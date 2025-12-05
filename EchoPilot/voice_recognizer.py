import speech_recognition as sr
import logging

def listen_for_command(timeout=5, phrase_time_limit=10):
    """
    Listens for a voice command and returns it as text.
    
    Args:
        timeout: Maximum time to wait for audio input (seconds)
        phrase_time_limit: Maximum length of audio phrase to record (seconds)
    
    Returns:
        str: Lowercased command text, or None if recognition failed
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("\n" + "="*25)
            print("🎤 Calibrating... Please be quiet for a moment.")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception as e:
                logging.warning(f"Could not adjust for ambient noise: {e}")
                # Continue anyway - this is not critical
            
            print("Say your command now...")
            print("="*25)
            
            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                print("✅ Audio captured, recognizing...")
                
                try:
                    command = recognizer.recognize_google(audio)
                    print(f"👤 YOU SAID: {command}")
                    return command.lower() if command else None
                except sr.UnknownValueError:
                    print("❌ Could not understand the audio.")
                    return None
                except sr.RequestError as e:
                    logging.error(f"Recognition service error: {e}")
                    print("❌ Speech recognition service unavailable.")
                    return None
                    
            except sr.WaitTimeoutError:
                print("👂 Listening timed out.")
                return None
            except Exception as e:
                logging.error(f"Unexpected error during audio capture: {e}")
                print(f"❌ Error: {e}")
                return None
                
    except Exception as e:
        logging.error(f"Fatal error in voice recognition: {e}")
        print(f"❌ Fatal error: {e}")
        return None