"""
EchoPilot Drone Agent - Intelligent Mission Planning and Execution System

This module provides an AI-powered drone control agent that:
- Listens to voice commands
- Plans missions using LLM reasoning
- Executes missions step-by-step with safety checks
- Provides real-time feedback via voice and console
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from operator import add

from langchain_core.messages import ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from speaker import speaker
from voice_recognizer import listen_for_command

# ==============================================================================
# == Configuration and Constants
# ==============================================================================

# Load environment variables from .env file
# Try multiple locations: current dir, parent dir, and default
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_paths = [
    os.path.join(current_dir, ".env"),
    os.path.join(parent_dir, ".env"),
    ".env",  # Default location (current working directory)
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)
        break
else:
    # If no .env found, still try default load_dotenv() behavior
    load_dotenv(override=False)

# Constants
DEFAULT_TAKEOFF_ALTITUDE = 20.0
DEFAULT_ORBIT_RADIUS = 50.0
DEFAULT_ORBIT_VELOCITY = 5.0
DEFAULT_FLIGHT_VELOCITY = 5.0
MAX_PLANNING_RETRIES = 3
JSON_EXTRACTION_TIMEOUT = 30.0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# == Type Definitions
# ==============================================================================

class MissionState(TypedDict):
    """State structure for the mission execution graph."""
    user_prompt: str
    mission_plan: List[Dict[str, Any]]
    messages: Annotated[List[BaseMessage], add]
    current_step_index: int
    target_location_details: Dict[str, Any]
    tool_schemas: str
    mission_status: str  # "planning", "executing", "completed", "failed"
    error_message: Optional[str]

# ==============================================================================
# == LLM Initialization
# ==============================================================================

def initialize_llm():
    """
    Initialize the Grok-3 LLM model.
    
    Requires XAI_API_KEY to be set in environment or .env file.
    The .env file should be loaded already by the module-level load_dotenv() call.
    """
    # Get API key from environment (should be loaded from .env or system env)
    api_key = os.getenv("XAI_API_KEY")
    
    if not api_key:
        error_msg = (
            "XAI_API_KEY is required but not found.\n"
            "Please set it in one of the following ways:\n"
            "  1. Environment variable: export XAI_API_KEY='your-key-here'\n"
            "  2. .env file: Create a .env file with XAI_API_KEY=your-key-here\n"
            "     (Place it in: EchoPilot/EchoPilot/.env or EchoPilot/.env)\n"
            "  3. System environment: Set it in your shell profile"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Set the API key in environment for langchain
    os.environ["XAI_API_KEY"] = api_key
    
    try:
        logger.info("Initializing Grok-3 model...")
        llm = init_chat_model(model="grok-3", model_provider="xai")
        logger.info("✅ Grok-3 model initialized successfully")
        return llm
    except Exception as e:
        error_msg = f"Failed to initialize Grok-3 model: {e}\nPlease check your XAI_API_KEY is valid."
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

LLM = initialize_llm()

# ==============================================================================
# == Helper Functions
# ==============================================================================

def extract_json_from_string(text: str) -> List[Dict[str, Any]]:
    """
    Robustly extracts a JSON list from a string with multiple fallback strategies.
    
    Args:
        text: The text containing JSON
        
    Returns:
        Parsed JSON list or empty list on failure
    """
    if not text or not isinstance(text, str):
        return []
    
    # Strategy 1: Look for markdown code blocks
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        # Strategy 2: Find first '[' and last ']'
        start_index = text.find('[')
        end_index = text.rfind(']')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = text[start_index:end_index + 1]
        else:
            return []
    
    # Try parsing with multiple strategies
    for attempt in range(2):
        try:
            parsed = json.loads(json_str)
            # Handle double-encoded JSON
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            
            # Validate it's a list
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and "plan" in parsed:
                return parsed["plan"]
            else:
                logger.warning(f"JSON is not a list: {type(parsed)}")
                return []
        except json.JSONDecodeError as e:
            if attempt == 0:
                # Try cleaning the string
                json_str = re.sub(r'[^\x20-\x7E\n\r\t]', '', json_str)
                continue
            logger.error(f"Failed to decode JSON: {e}\nSnippet: {json_str[:200]}")
            return []
    
    return []


def format_tools_for_prompt(tools: List[BaseTool]) -> str:
    """
    Formats tool schemas into a readable prompt format.
    
    Args:
        tools: List of available tools
        
    Returns:
        Formatted string describing all tools
    """
    schemas = []
    for tool in tools:
        try:
            schema = tool.get_input_schema().schema()
            properties = schema.get('properties', {})
            params = ", ".join([
                f"{name}: {props.get('type', 'any')}"
                for name, props in properties.items()
            ])
            description = tool.description or "No description available"
            schemas.append(f"- {tool.name}({params}): {description}")
        except Exception as e:
            logger.warning(f"Failed to format tool {tool.name}: {e}")
            schemas.append(f"- {tool.name}: {tool.description or 'No description'}")
    
    return "\n".join(schemas) if schemas else "No tools available"


def validate_mission_plan(plan: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """
    Validates a mission plan structure.
    
    Args:
        plan: The mission plan to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(plan, list):
        return False, "Mission plan must be a list"
    
    if len(plan) == 0:
        return False, "Mission plan is empty"
    
    required_tools = {"pre_flight_check", "arm_and_takeoff"}
    found_tools = {step.get("tool") for step in plan if isinstance(step, dict)}
    
    # Check for required initial steps
    if plan[0].get("tool") != "pre_flight_check":
        return False, "Mission plan must start with 'pre_flight_check'"
    
    # Validate each step structure
    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            return False, f"Step {i+1} is not a dictionary"
        if "tool" not in step:
            return False, f"Step {i+1} missing 'tool' field"
        if "args" not in step:
            return False, f"Step {i+1} missing 'args' field"
        if not isinstance(step["args"], dict):
            return False, f"Step {i+1} 'args' must be a dictionary"
    
    return True, None


def get_step_description(step: Dict[str, Any]) -> str:
    """Generate a human-readable description of a mission step."""
    tool_name = step.get("tool", "unknown")
    args = step.get("args", {})
    
    descriptions = {
        "pre_flight_check": "Running pre-flight safety checks",
        "arm_and_takeoff": f"Taking off to {args.get('altitude_meters', DEFAULT_TAKEOFF_ALTITUDE)} meters",
        "get_coordinates_for_location": f"Finding coordinates for {args.get('location_name', 'location')}",
        "fly_to_coordinates": f"Flying to coordinates ({args.get('latitude', '?')}, {args.get('longitude', '?')})",
        "fly_relative": f"Moving relative: forward {args.get('forward_meters', 0)}m, right {args.get('right_meters', 0)}m",
        "do_orbit": f"Orbiting at {args.get('radius_meters', DEFAULT_ORBIT_RADIUS)}m radius",
        "land": "Landing at current position",
        "return_to_launch": "Returning to launch point",
    }
    
    return descriptions.get(tool_name, f"Executing {tool_name}")

# ==============================================================================
# == Graph Nodes
# ==============================================================================

def planner_node(state: MissionState) -> Dict[str, Any]:
    """
    Generates a mission plan from user command using LLM reasoning.
    
    Uses an improved prompt with better examples and validation.
    """
    logger.info("🧠 PLANNER NODE: Generating mission plan...")
    
    user_prompt = state.get("user_prompt", "")
    if not user_prompt:
        logger.error("No user prompt provided")
        return {
            "mission_plan": [],
            "mission_status": "failed",
            "error_message": "No user command provided"
        }
    
    prompt = f"""You are a meticulous, highly intelligent flight operations officer for an autonomous drone. 
Your single, critical purpose is to convert a user's freeform command into a **perfectly structured, error-free, and executable** JSON list of tool calls. You must adhere strictly to the reasoning process and tool definitions provided.

--- AVAILABLE TOOLS ---
{state['tool_schemas']}
--- END TOOLS ---

--- YOUR REASONING PROCESS ---
1. **Deconstruct User Intent:** Read the user's entire command to understand the overall mission goal. Identify every distinct action requested (e.g., takeoff, fly to a place, move relative, orbit, land, return).

2. **Analyze Tool Descriptions:** For each action, find the **single best tool** from the "AVAILABLE TOOLS" list by carefully reading its description.
   - If the command is "go home" or "come back", use `return_to_launch`.
   - If the command is "land here" or "land now", use `land`.
   - If the command involves a specific named location ("Eiffel Tower", "home depot"), you MUST use `get_coordinates_for_location` first, followed by `fly_to_coordinates`.
   - If the command involves relative movement ("go forward 10 meters", "move up 5 meters", "go left 20 meters"), you MUST use `fly_relative`. DO NOT use `fly_to_coordinates` for relative movement.

3. **Extract Parameters & Apply Defaults:** Identify all explicit parameters from the user's command. If a required argument is missing, supply a safe default:
   - Default `altitude_meters` for `arm_and_takeoff` is {DEFAULT_TAKEOFF_ALTITUDE}.
   - Default `radius_meters` for `do_orbit` is {DEFAULT_ORBIT_RADIUS}. Default `velocity_ms` is {DEFAULT_ORBIT_VELOCITY}.
   - `fly_to_coordinates` and `do_orbit` can have optional `velocity_ms`. If the user specifies speed, include it. Otherwise, omit it.

4. **Sanitize Location Names:** Correct obvious spelling mistakes in location names (e.g., "eifeel tower" → "Eiffel Tower").

5. **Use Placeholders for Dynamic Data:** For missions involving named locations, `fly_to_coordinates` and `do_orbit` MUST use placeholders "TARGET_LAT" and "TARGET_LON" for coordinates.

6. **Construct the Final JSON:** Build the plan as a JSON list. Ensure:
   - `pre_flight_check` is always first
   - Every step has correct tool name and complete `args` dictionary
   - Steps are logical and sequential

--- EXAMPLES ---

**User Command:** "takeoff, fly 50 meters forward, then return home"
**Response:**
```json
[
  {{"tool": "pre_flight_check", "args": {{}}}},
  {{"tool": "arm_and_takeoff", "args": {{"altitude_meters": {DEFAULT_TAKEOFF_ALTITUDE}}}}},
  {{"tool": "fly_relative", "args": {{"forward_meters": 50}}}},
  {{"tool": "return_to_launch", "args": {{}}}}
]
```

**User Command:** "takeoff to 30m, fly to the Eiffel Tower at 15 m/s, circle it, then land"
**Response:**
```json
[
  {{"tool": "pre_flight_check", "args": {{}}}},
  {{"tool": "arm_and_takeoff", "args": {{"altitude_meters": 30}}}},
  {{"tool": "get_coordinates_for_location", "args": {{"location_name": "Eiffel Tower"}}}},
  {{"tool": "fly_to_coordinates", "args": {{"latitude": "TARGET_LAT", "longitude": "TARGET_LON", "velocity_ms": 15}}}},
  {{"tool": "do_orbit", "args": {{"latitude": "TARGET_LAT", "longitude": "TARGET_LON", "radius_meters": {DEFAULT_ORBIT_RADIUS}, "velocity_ms": 15}}}},
  {{"tool": "land", "args": {{}}}}
]
```
--- END EXAMPLES ---

Now, generate the complete and executable JSON plan for the following user command. Respond ONLY with the JSON list inside a markdown code block.

User Command: "{user_prompt}"
"""
    
    try:
        response = LLM.invoke(prompt)
        mission_plan = extract_json_from_string(response.content)
        
        if not mission_plan:
            logger.error("LLM failed to generate valid mission plan")
            logger.debug(f"LLM Raw Output:\n{response.content}")
            return {
                "mission_plan": [],
                "mission_status": "failed",
                "error_message": "Failed to generate valid mission plan"
            }
        
        # Validate the plan
        is_valid, error_msg = validate_mission_plan(mission_plan)
        if not is_valid:
            logger.error(f"Mission plan validation failed: {error_msg}")
            return {
                "mission_plan": [],
                "mission_status": "failed",
                "error_message": f"Invalid mission plan: {error_msg}"
            }
        
        logger.info(f"✅ Generated Mission Plan ({len(mission_plan)} steps):")
        for i, step in enumerate(mission_plan, 1):
            logger.info(f"  {i}. {get_step_description(step)}")
        
        return {
            "mission_plan": mission_plan,
            "mission_status": "executing"
        }
        
    except Exception as e:
        logger.error(f"Error in planner node: {e}", exc_info=True)
        return {
            "mission_plan": [],
            "mission_status": "failed",
            "error_message": f"Planning error: {str(e)}"
        }


def prepare_tool_call_node(state: MissionState) -> Dict[str, Any]:
    """
    Prepares a tool call for execution, handling coordinate placeholders.
    
    This node is robust and can inject coordinates if the LLM forgot placeholders.
    """
    plan = state.get("mission_plan", [])
    index = state.get("current_step_index", 0)
    
    if index >= len(plan):
        logger.error(f"Step index {index} out of range for plan with {len(plan)} steps")
        return {}
    
    step = plan[index]
    tool_name = step.get("tool")
    tool_args = step.get("args", {}).copy()
    
    if not tool_name:
        logger.error(f"Step {index} missing tool name")
        return {}
    
    # Handle coordinate placeholders
    target_location = state.get("target_location_details", {})
    if target_location and tool_name in ["fly_to_coordinates", "do_orbit"]:
        # Replace placeholders or inject coordinates
        if "latitude" in tool_args and tool_args["latitude"] in ["TARGET_LAT", "TARGET_LON"]:
            tool_args["latitude"] = target_location["latitude"]
        elif "latitude" not in tool_args:
            tool_args["latitude"] = target_location["latitude"]
        
        if "longitude" in tool_args and tool_args["longitude"] in ["TARGET_LAT", "TARGET_LON"]:
            tool_args["longitude"] = target_location["longitude"]
        elif "longitude" not in tool_args:
            tool_args["longitude"] = target_location["longitude"]
    
    total_steps = len(plan)
    progress = int((index + 1) / total_steps * 100) if total_steps > 0 else 0
    
    logger.info(f"⚙️ Step {index + 1}/{total_steps} ({progress}%): {get_step_description(step)}")
    logger.debug(f"Tool: {tool_name}, Args: {tool_args}")
    
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "id": f"call_{index}_{tool_name}",
                    "name": tool_name,
                    "args": tool_args
                }]
            )
        ]
    }


def decide_next_step_node(state: MissionState) -> Dict[str, Any]:
    """
    Decides the next step after tool execution, with safety checks.
    
    Handles errors, stores location data, and advances the step counter.
    """
    messages = state.get("messages", [])
    if not messages:
        logger.warning("No messages in state")
        return {"current_step_index": state.get("current_step_index", 0) + 1}
    
    last_message = messages[-1]
    updates = {}
    
    # Check for errors
    if isinstance(last_message, ToolMessage):
        is_error = (
            getattr(last_message, 'status', None) == 'error' or
            '"status": "Error"' in last_message.content
        )
        
        if is_error:
            error_msg = f"Mission failed at step {state.get('current_step_index', 0) + 1}"
            try:
                error_data = json.loads(last_message.content)
                error_msg += f": {error_data.get('message', last_message.content)}"
            except:
                error_msg += f": {last_message.content}"
            
            logger.error(f"❌ CRITICAL ERROR: {error_msg}")
            return {
                "mission_status": "failed",
                "error_message": error_msg,
                "current_step_index": state.get("current_step_index", 0) + 1
            }
        
        # Store location details from geocoding
        if last_message.name == "get_coordinates_for_location":
            try:
                result = json.loads(last_message.content)
                if result.get("status") == "Success":
                    logger.info(f"✅ Stored location: {result.get('address', 'Unknown')}")
                    updates["target_location_details"] = result
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse location details: {e}")
    
    # Preserve existing location details
    if "target_location_details" in state and state["target_location_details"]:
        updates["target_location_details"] = state["target_location_details"]
    
    # Advance step counter
    new_index = state.get("current_step_index", 0) + 1
    updates["current_step_index"] = new_index
    
    logger.info(f"✅ Step completed. Progress: {new_index}/{len(state.get('mission_plan', []))}")
    
    return updates


def should_plan_or_end(state: MissionState) -> str:
    """Router: Determines if planning succeeded and we should proceed."""
    plan = state.get("mission_plan", [])
    if plan and len(plan) > 0:
        return "prepare_tool_call"
    else:
        logger.warning("No mission plan generated. Ending.")
        return END


def should_continue_or_end(state: MissionState) -> str:
    """Router: Determines if mission should continue or end."""
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        is_error = (
            getattr(last_message, 'status', None) == 'error' or
            '"status": "Error"' in getattr(last_message, 'content', '')
        )
        if is_error:
            return END
    
    current_index = state.get("current_step_index", 0)
    plan = state.get("mission_plan", [])
    
    if current_index >= len(plan):
        logger.info("🎉 Mission plan fully executed.")
        return END
    else:
        return "prepare_tool_call"

# ==============================================================================
# == Input Interface Functions
# ==============================================================================

def get_text_command() -> Optional[str]:
    """
    Gets a command from text input (stdin).
    
    Returns:
        str: User command text, or None if input is empty/EOF
    """
    try:
        print("\n" + "="*60)
        print("📝 Enter your mission command (or 'quit' to exit):")
        print("="*60)
        command = input("> ").strip()
        
        if not command:
            return None
        
        print(f"✅ Command received: {command}")
        return command.lower()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Exiting...")
        return "quit"
    except Exception as e:
        logger.error(f"Error reading text input: {e}")
        return None


# ==============================================================================
# == Main Mission Execution
# ==============================================================================

async def run_mission(interface_mode: str = "voice"):
    """
    Main mission execution loop with voice or text interface.
    
    Args:
        interface_mode: "voice" for voice commands, "text" for text input
    
    Handles:
    - MCP server connection
    - Graph compilation
    - Command input (voice or text)
    - Mission execution with real-time feedback
    """
    logger.info("Initializing EchoPilot Drone Agent...")
    
    # Connect to MCP server and get tools
    try:
        # Get the current directory and Python executable
        current_dir = os.path.dirname(os.path.abspath(__file__))
        python_exe = sys.executable
        
        logger.info(f"Connecting to MCP server using Python: {python_exe}")
        logger.info(f"Server script path: {os.path.join(current_dir, 'drone_server.py')}")
        
        client = MultiServerMCPClient({
            "PX4DroneControlServer": {
                "command": python_exe,
                "args": [os.path.join(current_dir, "drone_server.py")],
                "transport": "stdio",
                "cwd": current_dir
            }
        })
        logger.info("Loading tools from MCP server...")
        
        # Add timeout to prevent hanging if MCP server takes too long
        try:
            tools = await asyncio.wait_for(client.get_tools(), timeout=15.0)
        except asyncio.TimeoutError:
            raise TimeoutError(
                "Timeout waiting for MCP server to provide tools. "
                "The server may be stuck connecting to the drone. "
                "Please check that drone_server.py is running correctly."
            )
        
        if not tools:
            raise ValueError("No tools loaded from MCP server")
        
        executor_node = ToolNode(tools)
        logger.info(f"✅ Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {e}", exc_info=True)
        await asyncio.to_thread(
            speaker.say,
            "Failed to connect to drone server. Please check that all dependencies are installed."
        )
        print(f"\n❌ MCP Connection Error Details:")
        print(f"   Error: {e}")
        print(f"   Python: {sys.executable}")
        print(f"   Working Directory: {current_dir}")
        print(f"   Server Script: {os.path.join(current_dir, 'drone_server.py')}")
        return
    
    # Build and compile the workflow graph
    workflow = StateGraph(MissionState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("prepare_tool_call", prepare_tool_call_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("decide_next_step", decide_next_step_node)
    
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner", should_plan_or_end)
    workflow.add_edge("prepare_tool_call", "executor")
    workflow.add_edge("executor", "decide_next_step")
    workflow.add_conditional_edges("decide_next_step", should_continue_or_end)
    
    app = workflow.compile()
    logger.info("✅ Mission workflow compiled successfully")
    
    # Greet user based on interface mode
    if interface_mode == "text":
        print("\n" + "="*60)
        print("🚁 EchoPilot Drone Agent - Text Interface")
        print("="*60)
        print("Ready for mission commands. Type 'quit' or 'exit' to shutdown.")
        print()
    else:
        await asyncio.to_thread(speaker.say, "EchoPilot is ready. Please state your mission command.")
    
    mission_count = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3
    
    # Main command loop
    while True:
        try:
            # Get command based on interface mode
            if interface_mode == "text":
                user_command = get_text_command()
            else:
                user_command = await asyncio.to_thread(listen_for_command)
            
            if not user_command:
                if interface_mode == "text":
                    # For text mode, empty input just means try again
                    print("⚠️  Empty command. Please enter a mission command.")
                    continue
                else:
                    # Voice mode handling
                    consecutive_failures += 1
                    
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        await asyncio.to_thread(
                            speaker.say,
                            "I'm having trouble hearing you. Please check your microphone and try again in a moment."
                        )
                        logger.warning(f"Multiple consecutive recognition failures ({consecutive_failures})")
                        # Reset counter and wait longer before next attempt
                        consecutive_failures = 0
                        await asyncio.sleep(2)  # Give user time to adjust
                    else:
                        await asyncio.to_thread(speaker.say, "I didn't catch that. Please try again.")
                        await asyncio.sleep(0.5)  # Brief pause before retry
                    continue
            
            # Reset failure counter on successful recognition
            consecutive_failures = 0
            
            # Handle exit commands
            if any(word in user_command for word in ["quit", "exit", "shutdown", "stop"]):
                if interface_mode == "text":
                    print("\n👋 Shutting down EchoPilot. Goodbye.")
                else:
                    await asyncio.to_thread(speaker.say, "Shutting down EchoPilot. Goodbye.")
                logger.info("User requested shutdown")
                break
            
            mission_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"MISSION #{mission_count}: {user_command}")
            logger.info(f"{'='*60}")
            
            if interface_mode == "text":
                print(f"\n🧠 Planning mission #{mission_count}...")
            else:
                await asyncio.to_thread(speaker.say, f"Understood. Planning mission number {mission_count} now.")
            
            # Initialize mission state
            initial_state: MissionState = {
                "user_prompt": user_command,
                "current_step_index": 0,
                "messages": [],
                "target_location_details": {},
                "tool_schemas": format_tools_for_prompt(tools),
                "mission_plan": [],
                "mission_status": "planning",
                "error_message": None
            }
            
            # Execute mission
            mission_successful = True
            executed_steps = []
            final_state = initial_state
            
            try:
                async for event in app.astream(initial_state):
                    for node_name, node_output in event.items():
                        logger.debug(f"Node '{node_name}' executed")
                        
                        # Update final state with latest node output
                        if isinstance(node_output, dict):
                            final_state.update(node_output)
                        
                        # Handle executor node output (tool execution results)
                        if node_name == "executor" and "messages" in node_output:
                            last_message = node_output["messages"][-1]
                            
                            if isinstance(last_message, ToolMessage):
                                # Check for errors
                                is_error = (
                                    getattr(last_message, 'status', None) == 'error' or
                                    '"status": "Error"' in last_message.content
                                )
                                
                                if is_error:
                                    mission_successful = False
                                    try:
                                        error_data = json.loads(last_message.content)
                                        error_msg = error_data.get("message", "Unknown error")
                                    except:
                                        error_msg = last_message.content
                                    
                                    if interface_mode == "text":
                                        print(f"❌ Error during {last_message.name}: {error_msg}")
                                    else:
                                        await asyncio.to_thread(
                                            speaker.say,
                                            f"Error during {last_message.name}. {error_msg}"
                                        )
                                else:
                                    # Success - provide feedback
                                    try:
                                        result_data = json.loads(last_message.content)
                                        status_msg = result_data.get("message", "completed")
                                        step_desc = get_step_description({
                                            "tool": last_message.name,
                                            "args": {}
                                        })
                                        if interface_mode == "text":
                                            print(f"✅ {step_desc}. {status_msg}")
                                        else:
                                            await asyncio.to_thread(
                                                speaker.say,
                                                f"{step_desc}. {status_msg}"
                                            )
                                        executed_steps.append({
                                            "tool": last_message.name,
                                            "status": "success",
                                            "message": status_msg
                                        })
                                    except json.JSONDecodeError:
                                        if interface_mode == "text":
                                            print(f"✅ {last_message.name} completed successfully.")
                                        else:
                                            await asyncio.to_thread(
                                                speaker.say,
                                                f"{last_message.name} completed successfully."
                                            )
                                        executed_steps.append({
                                            "tool": last_message.name,
                                            "status": "success"
                                        })
                
                # Check final state for mission status
                if final_state.get("mission_status") == "failed":
                    mission_successful = False
                    error_msg = final_state.get("error_message", "Unknown error")
                    if error_msg:
                        if interface_mode == "text":
                            print(f"❌ Mission failed: {error_msg}")
                        else:
                            await asyncio.to_thread(speaker.say, f"Mission failed. {error_msg}")
                
            except Exception as e:
                logger.error(f"Error during mission execution: {e}", exc_info=True)
                mission_successful = False
                if interface_mode == "text":
                    print(f"❌ An unexpected error occurred: {str(e)}")
                else:
                    await asyncio.to_thread(
                        speaker.say,
                        f"An unexpected error occurred: {str(e)}"
                    )
            
            # Mission summary
            if mission_successful:
                if interface_mode == "text":
                    print(f"\n✅ Mission #{mission_count} completed successfully. All {len(executed_steps)} steps executed.\n")
                else:
                    await asyncio.to_thread(
                        speaker.say,
                        f"Mission {mission_count} completed successfully. All {len(executed_steps)} steps executed."
                    )
                logger.info(f"✅ Mission #{mission_count} completed successfully")
            else:
                if interface_mode == "text":
                    print(f"\n❌ Mission #{mission_count} was aborted due to an error.\n")
                else:
                    await asyncio.to_thread(
                        speaker.say,
                        f"Mission {mission_count} was aborted due to an error."
                    )
                logger.error(f"❌ Mission #{mission_count} failed")
            
            # Ready for next command
            if interface_mode == "text":
                print("Ready for the next mission command.\n")
            else:
                await asyncio.to_thread(speaker.say, "Ready for the next mission command.")
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            if interface_mode == "text":
                print("\n👋 Interrupted. Shutting down.")
            else:
                await asyncio.to_thread(speaker.say, "Interrupted. Shutting down.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            if interface_mode == "text":
                print(f"❌ An error occurred: {e}. Please try again.")
            else:
                await asyncio.to_thread(speaker.say, "An error occurred. Please try again.")
            # Add a delay to prevent rapid error loops
            await asyncio.sleep(1)
            # Reset failure counter to avoid compounding issues
            consecutive_failures = 0
    
    logger.info("EchoPilot shutdown complete")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="EchoPilot Drone Agent - Intelligent Mission Planning and Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with voice interface (default)
  python drone_agent.py
  python drone_agent.py --interface voice
  
  # Run with text interface
  python drone_agent.py --interface text
  python drone_agent.py -i text
        """
    )
    
    parser.add_argument(
        "-i", "--interface",
        choices=["voice", "text"],
        default="voice",
        help="Interface mode: 'voice' for voice commands (default), 'text' for text input"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_mission(interface_mode=args.interface))
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        print("\n👋 Shutdown requested by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}\nCheck logs for details.")


if __name__ == "__main__":
    main()
