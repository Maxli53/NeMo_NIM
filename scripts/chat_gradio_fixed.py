#!/usr/bin/env python3
"""
GPT-OSS-20B ChatGPT-like Web Interface using Gradio - FIXED VERSION
Handles channel markers correctly to show full responses

Features:
- Real-time streaming responses
- ChatGPT-like interface design
- Conversation history
- Model settings controls
- Export/import conversations
- Dark/light theme support
- FIXED: Proper channel handling for GPT-OSS

Author: Unsloth GPT Project
Date: 2025-10-05
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
import torch
import gradio as gr
from typing import List, Tuple, Generator, Optional
from threading import Thread
import queue

# Unsloth and transformers imports
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# Global model variables
model = None
tokenizer = None
model_loaded = False

class ChannelStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria that doesn't stop at intermediate <|end|> tokens"""
    def __init__(self, tokenizer, stop_at_final_only=True):
        self.tokenizer = tokenizer
        self.stop_at_final_only = stop_at_final_only
        # Get token IDs for markers
        self.return_token = tokenizer.convert_tokens_to_ids("<|return|>")
        self.end_token = tokenizer.convert_tokens_to_ids("<|end|>")
        self.final_marker = tokenizer.encode("final<|message|>", add_special_tokens=False)
        self.has_final = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Check if we've seen the final channel marker
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        if "final<|message|>" in generated_text:
            self.has_final = True

        # Only stop at <|return|> if we're in final channel
        last_token = input_ids[0][-1].item()
        if last_token == self.return_token and self.has_final:
            return True

        # Don't stop at intermediate <|end|> tokens
        if self.stop_at_final_only and last_token == self.end_token:
            return False

        return False

def load_model_once(model_path: str = "models/latest", gpu_id: int = 1):
    """Load model only once at startup"""
    global model, tokenizer, model_loaded

    if model_loaded:
        return True

    print("🚀 Loading GPT-OSS-20B model...")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        # Load base model first (4-bit quantized)
        print("Loading base model (4-bit quantized)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Apply LoRA adapter if using fine-tuned model
        if not model_path.startswith("unsloth/") and os.path.exists(model_path):
            print(f"Applying LoRA adapter from {model_path}...")
            model = PeftModel.from_pretrained(model, model_path)

        # Enable inference mode
        FastLanguageModel.for_inference(model)

        model_loaded = True
        print("✅ Model loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return False

def format_message_for_display(content: str, show_thinking: bool = True) -> str:
    """Format the model output for display, optionally hiding thinking process"""

    # Clean up any special tokens
    content = content.replace("<|startoftext|>", "")
    content = content.replace("<|endoftext|>", "")

    if not show_thinking:
        # Extract only the final response
        if "final<|message|>" in content:
            # Extract content after final channel marker
            parts = content.split("final<|message|>")
            if len(parts) > 1:
                final_content = parts[-1]
                # Clean up any end markers
                for marker in ["<|return|>", "<|end|>", "<|endoftext|>"]:
                    if marker in final_content:
                        final_content = final_content.split(marker)[0]
                return final_content.strip()

        # If no final channel, look for commentary
        if "commentary<|message|>" in content:
            parts = content.split("commentary<|message|>")
            if len(parts) > 1:
                comment_content = parts[-1]
                for marker in ["<|return|>", "<|end|>", "<|endoftext|>"]:
                    if marker in comment_content:
                        comment_content = comment_content.split(marker)[0]
                return comment_content.strip()

        # Fallback: Try to extract something meaningful
        # Remove analysis channel if present
        if "analysis<|message|>" in content:
            parts = content.split("<|end|>")
            # Try to find non-analysis content
            for part in parts[1:]:
                cleaned = part.strip()
                if cleaned and not cleaned.startswith("analysis"):
                    return cleaned

    # Show full content with formatting for thinking mode
    if show_thinking:
        # Format channels nicely
        content = content.replace("<|channel|>analysis<|message|>", "\n**[Analysis]**\n")
        content = content.replace("<|channel|>commentary<|message|>", "\n\n**[Commentary]**\n")
        content = content.replace("<|channel|>final<|message|>", "\n\n**[Final Response]**\n")
        content = content.replace("<|end|>", "\n---")
        content = content.replace("<|return|>", "")

    return content.strip()

def generate_response(
    message: str,
    history: List[Tuple[str, str]],
    temperature: float = 0.7,
    max_new_tokens: int = 500,
    top_p: float = 0.9,
    top_k: int = 50,
    reasoning_effort: str = "medium",
    show_thinking: bool = True
) -> Generator[str, None, None]:
    """Generate streaming response for chat interface"""

    if not model_loaded:
        yield "⚠️ Model not loaded. Please wait..."
        return

    # Build conversation messages
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

    # Add conversation history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    # Add current message
    messages.append({"role": "user", "content": message})

    # Apply chat template
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        # Fallback if reasoning_effort not supported
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Tokenize
    inputs = tokenizer([formatted], return_tensors="pt").to("cuda")

    # Setup streaming with custom stopping criteria
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        timeout=60,
    )

    # Use custom stopping criteria
    stopping_criteria = StoppingCriteriaList([
        ChannelStoppingCriteria(tokenizer, stop_at_final_only=True)
    ])

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": True,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        # Don't use default EOS, let our custom criteria handle it
        "eos_token_id": None,
        "stopping_criteria": stopping_criteria,
    }

    # Start generation in thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the response
    generated_text = ""
    channels_seen = {"analysis": False, "commentary": False, "final": False}
    in_final_channel = False
    final_channel_buffer = ""

    for new_text in streamer:
        generated_text += new_text

        # Track which channels we've seen
        if "analysis<|message|>" in generated_text and not channels_seen["analysis"]:
            channels_seen["analysis"] = True
        if "commentary<|message|>" in generated_text and not channels_seen["commentary"]:
            channels_seen["commentary"] = True
        if "final<|message|>" in generated_text and not channels_seen["final"]:
            channels_seen["final"] = True
            in_final_channel = True

        # If we're in the final channel, accumulate its content
        if in_final_channel:
            # Extract just the new part of the final channel
            if "final<|message|>" in new_text:
                # This token contains the marker, extract what comes after
                final_part = new_text.split("final<|message|>")[-1]
                final_channel_buffer += final_part
            else:
                # Regular content in final channel
                final_channel_buffer += new_text

        # Check if we should stop
        if "<|return|>" in new_text and channels_seen["final"]:
            # We've hit the return in final channel, clean up and yield final
            final_channel_buffer = final_channel_buffer.replace("<|return|>", "")
            if show_thinking:
                # Show everything formatted
                yield format_message_for_display(generated_text.replace("<|return|>", ""), show_thinking)
            else:
                # Show only the final channel content
                yield final_channel_buffer.strip()
            break

        # Handle intermediate streaming
        if show_thinking:
            # Show everything as it generates
            yield format_message_for_display(generated_text, show_thinking)
        elif in_final_channel:
            # Only show content once we're in the final channel
            clean_buffer = final_channel_buffer.replace("<|return|>", "").replace("<|end|>", "")
            if clean_buffer.strip():
                yield clean_buffer.strip()
        # else: Don't yield anything while in analysis/commentary channels

    thread.join()

    # Final cleanup if we didn't get a proper final channel
    if not channels_seen["final"] and channels_seen["analysis"]:
        # Model only generated analysis, try to extract something useful
        if show_thinking:
            yield format_message_for_display(generated_text, show_thinking)
        else:
            # Try to extract something meaningful from analysis
            yield "I apologize, but I couldn't generate a complete response. Please try again."

def create_interface():
    """Create the Gradio interface"""

    # Custom CSS for ChatGPT-like appearance
    custom_css = """
    #chatbot {
        height: 600px !important;
        overflow-y: auto;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }

    .message.user {
        background-color: #f7f7f8;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
    }

    .message.assistant {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
    }

    #input-box {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 10px;
        font-size: 16px;
    }

    #send-button {
        background-color: #10a37f;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }

    #send-button:hover {
        background-color: #0d8f6f;
    }

    .settings-panel {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        #chatbot {
            background-color: #1e1e1e;
            border-color: #444;
        }

        .message.user {
            background-color: #2a2a2a;
            color: #e0e0e0;
        }

        .message.assistant {
            background-color: #333333;
            color: #e0e0e0;
        }

        #input-box {
            background-color: #2a2a2a;
            color: #e0e0e0;
            border-color: #444;
        }

        .settings-panel {
            background-color: #2a2a2a;
            color: #e0e0e0;
        }
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        #chatbot {
            height: 400px !important;
        }
    }
    """

    # Create the interface
    with gr.Blocks(title="GPT-OSS-20B Chat", theme=gr.themes.Soft(), css=custom_css) as interface:

        # Header
        gr.Markdown(
            """
            # 🤖 GPT-OSS-20B Chat Interface (Fixed)
            ### Powered by Unsloth • Running at 15-16 tokens/sec on RTX 3090
            #### ✅ Fixed: Now shows complete responses from all channels
            """
        )

        with gr.Row():
            # Main chat area (left side, larger)
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    show_label=False,
                    height=600,
                    avatar_images=["🧑‍💻", "🤖"],
                    render_markdown=True,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here... (Press Enter to send)",
                        lines=2,
                        elem_id="input-box",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", elem_id="send-button", scale=1, variant="primary")

                with gr.Row():
                    clear = gr.Button("🗑️ Clear Chat", scale=1)
                    regenerate = gr.Button("🔄 Regenerate", scale=1)
                    stop = gr.Button("⏹️ Stop", scale=1)

            # Settings panel (right side)
            with gr.Column(scale=1, elem_classes="settings-panel"):
                gr.Markdown("### ⚙️ Settings")

                with gr.Accordion("Model Parameters", open=True):
                    temperature = gr.Slider(
                        minimum=0, maximum=2, value=0.7, step=0.1,
                        label="Temperature",
                        info="Higher = more creative"
                    )

                    max_tokens = gr.Slider(
                        minimum=50, maximum=2000, value=500, step=50,
                        label="Max Tokens",
                        info="Maximum response length"
                    )

                    top_p = gr.Slider(
                        minimum=0, maximum=1, value=0.9, step=0.05,
                        label="Top-p",
                        info="Nucleus sampling threshold"
                    )

                    top_k = gr.Slider(
                        minimum=0, maximum=100, value=50, step=5,
                        label="Top-k",
                        info="Top-k sampling (0 = disabled)"
                    )

                    reasoning_effort = gr.Radio(
                        choices=["low", "medium", "high"],
                        value="medium",
                        label="Reasoning Effort",
                        info="GPT-OSS thinking depth"
                    )

                    show_thinking = gr.Checkbox(
                        value=False,
                        label="Show Thinking Process",
                        info="Display model's reasoning channels"
                    )

                with gr.Accordion("Conversation", open=False):
                    export_btn = gr.Button("💾 Export Chat")
                    import_btn = gr.UploadButton("📂 Import Chat", file_types=["json"])

                    export_file = gr.File(
                        label="Exported Chat",
                        visible=False
                    )

                gr.Markdown(
                    """
                    ### 📊 Stats
                    - Model: GPT-OSS-20B (4-bit)
                    - VRAM: ~12.4GB
                    - Speed: 15-16 tokens/sec
                    - **Fixed**: Channel handling
                    """
                )

        # Footer
        gr.Markdown(
            """
            ---
            *Built with [Unsloth](https://github.com/unslothai/unsloth) and [Gradio](https://gradio.app)*
            *Fixed version handles GPT-OSS channel markers correctly*
            """
        )

        # Event handlers
        def user_submit(message, history):
            """Handle user message submission"""
            return "", history + [[message, None]]

        def bot_response(history, temp, max_tok, tp, tk, reasoning, thinking):
            """Generate bot response"""
            if not history or not history[-1][0]:
                yield history
                return

            user_message = history[-1][0]
            history[-1][1] = ""

            for partial_response in generate_response(
                user_message,
                history[:-1],
                temperature=temp,
                max_new_tokens=max_tok,
                top_p=tp,
                top_k=tk,
                reasoning_effort=reasoning,
                show_thinking=thinking
            ):
                history[-1][1] = partial_response
                yield history

        def clear_chat():
            """Clear the chat history"""
            return None

        def export_chat(history):
            """Export chat history to JSON"""
            if not history:
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_export_{timestamp}.json"

            export_data = {
                "timestamp": datetime.now().isoformat(),
                "model": "GPT-OSS-20B",
                "conversations": [
                    {"user": user, "assistant": assistant}
                    for user, assistant in history
                ]
            }

            export_path = Path(f"chat_exports/{filename}")
            export_path.parent.mkdir(exist_ok=True)

            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            return gr.File(value=str(export_path), visible=True)

        def import_chat(file):
            """Import chat history from JSON"""
            if not file:
                return None

            with open(file.name, 'r') as f:
                data = json.load(f)

            history = [
                [conv["user"], conv["assistant"]]
                for conv in data.get("conversations", [])
            ]

            return history

        # Wire up events
        msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response,
            [chatbot, temperature, max_tokens, top_p, top_k, reasoning_effort, show_thinking],
            chatbot
        )

        send_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response,
            [chatbot, temperature, max_tokens, top_p, top_k, reasoning_effort, show_thinking],
            chatbot
        )

        clear.click(clear_chat, None, chatbot, queue=False)

        export_btn.click(export_chat, chatbot, export_file)
        import_btn.upload(import_chat, import_btn, chatbot)

        # Regenerate last response
        def regenerate_last(history):
            if history and history[-1][1]:
                # Keep user message, clear assistant response
                history[-1][1] = None
                return history
            return history

        regenerate.click(regenerate_last, chatbot, chatbot, queue=False).then(
            bot_response,
            [chatbot, temperature, max_tokens, top_p, top_k, reasoning_effort, show_thinking],
            chatbot
        )

    return interface

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GPT-OSS-20B Web Chat Interface (Fixed)")
    parser.add_argument("--model_path", type=str, default="models/latest",
                       help="Path to model or 'models/latest' for most recent")
    parser.add_argument("--gpu", type=int, default=1,
                       help="GPU to use (0 or 1)")
    parser.add_argument("--share", action="store_true",
                       help="Share the interface publicly (generates a link)")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the server on")
    parser.add_argument("--server_name", type=str, default="127.0.0.1",
                       help="Server name (use 0.0.0.0 for network access)")

    args = parser.parse_args()

    # Load model first
    print("\n" + "="*60)
    print("GPT-OSS-20B Web Chat Interface (Fixed Version)")
    print("="*60)

    if not load_model_once(args.model_path, args.gpu):
        print("Failed to load model. Exiting...")
        sys.exit(1)

    # Create and launch interface
    interface = create_interface()

    print("\n🌐 Launching web interface...")
    print(f"📍 Local URL: http://{args.server_name}:{args.port}")

    if args.share:
        print("🔗 Generating public share link...")

    # Launch the interface
    interface.queue()  # Enable queuing for streaming
    interface.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
        quiet=False,
    )

if __name__ == "__main__":
    main()