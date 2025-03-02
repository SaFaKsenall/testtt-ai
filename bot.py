import os
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import telegram
import requests
from io import BytesIO
from deep_translator import GoogleTranslator
import google.generativeai as genai
import asyncio
import time
from gtts import gTTS
import speech_recognition as sr
import tempfile
import soundfile as sf
import json

# Load environment variables
load_dotenv()

# Configure logging with more detail
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Changed from INFO to DEBUG for more detail
)
logger = logging.getLogger(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Hugging Face API configuration
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# API URLs
IMAGE_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
SPEECH_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
TRANSLATION_API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tr-en"
VOICE_TRANSLATION_URL = "https://api-inference.huggingface.co/models/coqui/XTTS-v2"
MUSIC_GEN_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"

# User states dictionary to track what feature each user is using
user_states = {}

async def transcribe_audio(audio_bytes):
    """Transcribe audio using Google Speech Recognition"""
    try:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_ogg:
            temp_ogg.write(audio_bytes)
            temp_ogg.flush()
            
            # Convert OGG to WAV using soundfile
            data, samplerate = sf.read(temp_ogg.name)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                sf.write(temp_wav.name, data, samplerate)
                
                # Initialize recognizer
                recognizer = sr.Recognizer()
                
                # Load the audio file
                with sr.AudioFile(temp_wav.name) as source:
                    # Record the audio
                    audio = recognizer.record(source)
                    
                    # Recognize speech using Google Speech Recognition
                    text = recognizer.recognize_google(audio, language="tr-TR")
                    
                    return text
                    
    except Exception as e:
        logger.error(f"Speech recognition error: {str(e)}")
        raise e
    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_ogg.name)
            os.unlink(temp_wav.name)
        except:
            pass

async def translate_text(text):
    """Translate text using Google Translate"""
    try:
        translator = GoogleTranslator(source='tr', target='en')
        translated = translator.translate(text)
        logger.debug(f"Original text: {text}")
        logger.debug(f"Translated text: {translated}")
        return translated
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise e

async def text_to_speech(text):
    """Convert text to speech using Google Text-to-Speech"""
    try:
        # Create a BytesIO buffer to store the audio
        audio_buffer = BytesIO()
        
        # Create gTTS object and save to buffer
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(audio_buffer)
        
        # Reset buffer position
        audio_buffer.seek(0)
        return audio_buffer
        
    except Exception as e:
        logger.error(f"Text to speech error: {str(e)}")
        raise e

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages"""
    try:
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🎤 Your voice message is being processed...\n⏳ Estimated time: 10 seconds"
        )
        
        # Get voice message file
        voice = await context.bot.get_file(update.message.voice.file_id)
        
        # Download voice file
        voice_file = BytesIO()
        await voice.download_to_memory(voice_file)
        
        # Start transcription countdown
        countdown_task = asyncio.create_task(
            countdown_message(
                processing_msg,
                10,
                "🎤 Your voice message is being converted...",
                "Processing"
            )
        )
        
        # Transcribe audio
        recognized_text = await transcribe_audio(voice_file.getvalue())
        
        # Translate text
        translated_text = await translate_text(recognized_text)
        
        # Wait for countdown
        await countdown_task
        
        # Send results
        await update.message.reply_text(
            f"🎤 Your voice message has been converted:\n\n"
            f"📝 original text:\n{recognized_text}\n\n"
            f"🔄 Translation:\n{translated_text}"
        )
        
        # Delete processing message
        await processing_msg.delete()
        
    except Exception as e:
        logger.error(f"Voice handling error: {str(e)}")
        await update.message.reply_text(
            "Sorry, there was an error processing your voice message. Please try again."
        )

async def countdown_message(message, seconds, current_text, phase_name):
    """Show countdown message"""
    try:
        remaining = seconds
        while remaining > 0:
            try:
                await message.edit_text(
                    f"{current_text}\n\n"
                    f"⏳ {phase_name} remaining time for: {remaining} Second"
                )
            except telegram.error.BadRequest as e:
                if "Message to edit not found" in str(e):
                    logger.debug("Countdown message was deleted, stopping countdown")
                    break
                logger.error(f"Bad request while updating countdown: {str(e)}")
                break
            except Exception as e:
                logger.error(f"Error updating countdown: {str(e)}")
                break
            
            await asyncio.sleep(1)
            remaining -= 1
        return True
    except Exception as e:
        logger.error(f"Countdown message error: {str(e)}")
        return False

async def enhance_prompt_with_ai(prompt):
    """Use Gemini AI to enhance the prompt for better image generation"""
    try:
        system_prompt = """Sen bir görsel istek çevirmensin. Verilen Türkçe metni Stable Diffusion için net ve detaylı bir İngilizce prompt'a dönüştür.

Önemli kurallar:
- Orijinal prompt'taki TÜM nesneleri ve öğeleri koru
- Belirtilen tüm özellikleri (renkler, boyutlar, sayılar) koru
- Ana konunun doğru şekilde belirtildiğinden emin ol
- Netlik için minimal ama yardımcı detaylar ekle
- Basit, doğrudan bir açıklama olarak formatla

Örnek girdi: "Siyah gözlü 2 metre kuyruklu at"
Örnek çıktı: "A horse with black eyes and a 2-meter long tail, highly detailed"

Sadece çevrilmiş prompt'u döndür, açıklama yapma."""
        
        response = await model.generate_content_async(f"{system_prompt}\n\nInput: {prompt}")
        enhanced = response.text.strip()
        logger.debug(f"Original prompt: {prompt}")
        logger.debug(f"AI enhanced prompt: {enhanced}")
        return enhanced
    except Exception as e:
        logger.error(f"AI enhancement error: {str(e)}")
        # If enhancement fails, return a basic translation
        translator = GoogleTranslator(source='tr', target='en')
        try:
            return translator.translate(prompt)
        except:
            return prompt  # Fallback to original prompt

def enhance_prompt(prompt):
    """Add better enhancements to the prompt"""
    # Add more specific qualifiers that guide the model better
    enhancements = [
        "highly detailed",
        "sharp focus", 
        "professional photography",
        "realistic",
        "4k"
    ]
    return f"{prompt}, {', '.join(enhancements)}"

async def update_processing_message(message, start_time, current_text):
    """Update the processing message with elapsed time"""
    while True:
        elapsed = int(time.time() - start_time)
        await message.edit_text(f"{current_text}\n\n⏱️ elapsed time: {elapsed} Second")
        await asyncio.sleep(1)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message with feature selection buttons"""
    keyboard = [
        [
            InlineKeyboardButton("🎨 Text → Picture", callback_data='image_gen'),
            InlineKeyboardButton("🎤 Voice → Text", callback_data='voice_text')
        ],
        [
            InlineKeyboardButton("🗣️ Voice Translation", callback_data='voice_translation'),
            InlineKeyboardButton("❓ How to Use", callback_data='help')
        ],
        [
            InlineKeyboardButton("🐦 Twitter", url='https://x.com/alphackai')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_message = f"""
Hello, {update.effective_user.username or 'guest'}! I am a very talented AI bot! 🤖\n\nPlease select the feature you want to use:
"""
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button presses"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    if query.data == 'image_gen':
        user_states[user_id] = 'image_gen'
        text = """🎨 Text → Image feature selected!\n\nHow to Use:\n1. Write in detail the image you want to create.\n\nExample: "A black horse walking on the beach at sunset"\n\nLet's get started! What kind of image would you like to create??\n\nCheck Out Our Other Artificial Intelligence Bots: /start"""
        
    elif query.data == 'voice_text':
        user_states[user_id] = 'voice_text'
        text = """🎤 Voice → Text feature selected!\n\nHow to Use:\n1. Send me a voice message\n\nLet's get started! Send a voice message.\n\nCheck Out Our Other Artificial Intelligence Bots: /start"""
        
    elif query.data == 'voice_translation':
        user_states[user_id] = 'voice_translation'
        text = """🗣️ Voice Translation feature selected!\n\nHow to Use:\n1. Send me a voice message\n2. First, I will convert your message to text\n3. Then I will translate it into English\n4. Finally, I will send you the English voice message\n\nYou will also be able to see both Turkish and English texts.\n\nLet's get started! Send a voice message.\n\nCheck Out Our Other AI Bots: /start"""
        
    elif query.data == 'help':
        text = """
        ❓ Bot User Guide\n\n        🎨 Text → Picture:\n        • Identify the image you want\n        • The more detailed you define, the better the result will be\n        • Image rendering takes about 20-30 seconds\n\n        🎤 Voice → Text:\n        • Send any voice message\n        • You will receive both the original text and the English translation\n        • The process takes approximately 10-15 seconds\n\n        🗣️ Voice Translation:\n        • Send a voice message in Turkish\n        • You will receive both written and audio translations\n        • The process takes approximately 15-20 seconds\n\n        ❓ How to Use:\n        • You can use the /start command for usage instructions..\n        """

    await query.edit_message_text(text=text)

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages based on user's selected feature"""
    user_id = update.effective_user.id
    
    # If user types '/', redirect to homepage
    if update.message.text == '/':
        await homepage(update, context)
        return
    
    # If user hasn't selected a feature, prompt them to do so
    if user_id not in user_states:
        await update.message.reply_text(
            "Please select the feature you want to use first.\nYou can use the /start command for this."
        )
        return
    
    # Handle message based on selected feature
    if user_states[user_id] == 'image_gen':
        if update.message.text:
            await generate_image(update, context)
        else:
            await update.message.reply_text(
                "Please send a text to create an image."
            )
    
    elif user_states[user_id] == 'voice_text':
        if update.message.voice:
            await handle_voice(update, context)
        else:
            await update.message.reply_text(
                "Please send a voice message."
            )
    
    elif user_states[user_id] == 'voice_translation':
        if update.message.voice:
            await handle_voice_translation(update, context)
        else:
            await update.message.reply_text(
                "Please send a voice message."
            )
    
    elif user_states[user_id] == 'music_gen':
        if update.message.text:
            await handle_music_gen(update, context)
        else:
            await update.message.reply_text(
                "Please write a description to create music."
            )

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Get the text message from user
        text_prompt = update.message.text
        start_time = time.time()
        
        # Send initial processing message
        processing_message = await update.message.reply_text("🤖 Your photo is being prepared...")
        
        # Start the timer update task
        timer_task = asyncio.create_task(
            update_processing_message(
                processing_message,
                start_time,
                "🤖 Your request is being prepared..."
            )
        )
        
        # Enhance and translate the prompt
        ai_enhanced_prompt = await enhance_prompt_with_ai(text_prompt)
        final_prompt = enhance_prompt(ai_enhanced_prompt)
        
        # Update message with the enhanced prompt
        current_text = (
            f"🎨 Your image is being created...\n\n"
            f"Original: '{text_prompt}'\n"
            f"Developed: '{ai_enhanced_prompt}'"
        )
        
        # Update the timer task with new text
        timer_task.cancel()
        timer_task = asyncio.create_task(
            update_processing_message(
                processing_message,
                start_time,
                current_text
            )
        )
        
        # Better negative prompt
        negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, wrinkles, watermark, signature, text, low quality, worst quality"
        
        # Generate image using Stable Diffusion XL with better parameters
        response = requests.post(
            IMAGE_API_URL,
            headers=HEADERS,
            json={
                "inputs": final_prompt,
                "parameters": {
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": 50,  # Increased from 40
                    "guidance_scale": 8.5,      # Increased from 7.5
                    "width": 768,               # Add specific dimensions
                    "height": 768
                }
            }
        )
        
        if response.status_code == 503:
            # Model is loading, update message and wait
            await processing_message.edit_text(f"{current_text}\n\n⏳ Model is loading, please wait...")
            # Wait and retry
            await asyncio.sleep(20)
            response = requests.post(
                IMAGE_API_URL,
                headers=HEADERS,
                json={
                    "inputs": final_prompt,
                    "parameters": {
                        "negative_prompt": negative_prompt,
                        "num_inference_steps": 50,
                        "guidance_scale": 8.5,
                        "width": 768,
                        "height": 768
                    }
                }
            )
        
        if response.status_code != 200:
            error_content = response.content.decode('utf-8', errors='ignore')
            logger.error(f"Image API Error: {error_content}")
            raise Exception(f"API request failed with status {response.status_code}")
        
        # Stop the timer
        timer_task.cancel()
        
        # Calculate total time
        total_time = int(time.time() - start_time)
        
        # Get the image data directly from the response
        image_data = BytesIO(response.content)
        image_data.seek(0)
        
        # Send the image with time info and prompt details
        await update.message.reply_photo(
            photo=image_data,
            caption=f"✨ Picture {total_time} Created in seconds.\n\nExplanation: {ai_enhanced_prompt}"
        )
        
        # Delete the processing message
        await processing_message.delete()
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        await update.message.reply_text(
            "Sorry, there was an error creating the image. Please try again with a different description."
        )

async def handle_voice_translation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice translation messages"""
    try:
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🎤 Your voice message is being processed...\n⏳ Estimated time: 15 seconds"
        )
        
        # Get voice message file
        voice = await context.bot.get_file(update.message.voice.file_id)
        
        # Download voice file
        voice_file = BytesIO()
        await voice.download_to_memory(voice_file)
        
        # Start transcription countdown
        countdown_task = asyncio.create_task(
            countdown_message(
                processing_msg,
                15,
                "🎤 Your voice message is being translated...",
                "Processing"
            )
        )
        
        try:
            # Transcribe audio to Turkish text
            recognized_text = await transcribe_audio(voice_file.getvalue())
            
            # Translate text to English
            translated_text = await translate_text(recognized_text)
            
            # Convert English text to speech
            english_audio = await text_to_speech(translated_text)
            
            # Wait for countdown
            await countdown_task
            
            # Send results
            await update.message.reply_text(
                f"🎤 Your voice message has been converted:\n\n"
                f"📝 Türkish:\n{recognized_text}\n\n"
                f"🔄 English:\n{translated_text}"
            )
            
            # Send translated voice message
            await update.message.reply_voice(
                voice=english_audio,
                caption="🗣️ English voice message"
            )
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            await update.message.reply_text(
                "Your voice message was transcribed, but the voice message could not be created:\n\n"
                f"📝 Turkish-English:\n{recognized_text}\n\n"
                f"🔄 English:\n{translated_text}"
            )
        
        # Delete processing message
        await processing_msg.delete()
        
    except Exception as e:
        logger.error(f"Voice translation error: {str(e)}")
        await update.message.reply_text(
            "Sorry, there was an error translating the audio. Please try again."
        )

async def generate_music(prompt):
    """Generate music using MusicGen Small through Hugging Face API"""
    try:
        logger.debug(f"Attempting to generate music with prompt: {prompt}")
        logger.debug(f"Using MusicGen Small URL: {MUSIC_GEN_URL}")
        
        payload = {
            "inputs": f"Generate a melancholic melody: {prompt}",
            "parameters": {
                "wait_for_model": True
            }
        }
        
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        logger.debug(f"Request headers: {json.dumps(HEADERS, indent=2)}")
        
        response = requests.post(
            MUSIC_GEN_URL,
            headers=HEADERS,
            json=payload,
            timeout=60
        )
        
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 503:
            logger.warning("Model is loading, waiting and retrying...")
            # Wait for 20 seconds and try again
            time.sleep(20)
            response = requests.post(
                MUSIC_GEN_URL,
                headers=HEADERS,
                json=payload,
                timeout=60
            )
        
        if response.status_code != 200:
            error_content = response.content.decode('utf-8', errors='ignore')
            logger.error(f"API Error Response: {error_content}")
            raise Exception(f"API request failed with status code: {response.status_code}")
        
        # Convert response content to audio file
        audio_data = BytesIO(response.content)
        audio_data.seek(0)
        
        logger.debug("Successfully generated music and created audio buffer")
        return audio_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during music generation: {str(e)}")
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Music generation error: {str(e)}")
        raise

async def handle_music_gen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle music generation requests"""
    # Müzik oluşturma işlevselliği henüz uygulanmamış
    await update.message.reply_text(
        "The music creation feature is currently under development. Please try again later."
    )

async def homepage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message with feature selection buttons"""
    user_id = update.effective_user.id
    user_states.pop(user_id, None)  # Reset user state
    keyboard = [
        [
            InlineKeyboardButton("🎨 Text → Image", callback_data='image_gen'),
            InlineKeyboardButton("🎤 Voice → Text", callback_data='voice_text')
        ],
        [
            InlineKeyboardButton("🗣️ Voice Translation", callback_data='voice_translation'),
            InlineKeyboardButton("❓How to Use", callback_data='help')
        ],
        [
            InlineKeyboardButton("🐦 Twitter", url='https://x.com/alphackai')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_message = f"""
Merhaba, {update.effective_user.username or 'guest'}! I am a very talented AI bot! 🤖\n\nPlease select the feature you want to use:
"""
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

def main():
    """Start the bot"""
    # Create the Application and pass it your bot's token
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_TOKEN not found! Please check the .env file.")
    
    application = Application.builder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT | filters.VOICE, message_handler))
    application.add_handler(CommandHandler("homepage", homepage))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
