#!/usr/bin/env python3
"""
Interactive Chat Interface for GPT-OSS-20B using Unsloth
Full-featured terminal chat with streaming, history, and rich formatting

Features:
- Real-time token streaming (see response as it generates)
- Conversation history management
- Rich terminal formatting with colors
- Commands: /help, /clear, /save, /load, /stats, /exit
- Performance metrics display
- Markdown rendering support
- Session persistence

Author: Unsloth GPT Project
Date: 2025-10-04
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch

# Rich for beautiful terminal formatting
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.live import Live
    from rich.spinner import Spinner
    RICH_AVAILABLE = True
except ImportError:
    print("Warning: Rich not installed. Install with: pip install rich")
    print("Falling back to basic formatting.")
    RICH_AVAILABLE = False

# Unsloth and transformers
from unsloth import FastLanguageModel
from transformers import TextStreamer, TextIteratorStreamer
from threading import Thread
from peft import PeftModel

class ChatSession:
    """Manages chat history and session data"""

    def __init__(self, session_dir: str = "chat_sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.history: List[Dict[str, str]] = []
        self.stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "message_count": 0,
            "session_start": datetime.now()
        }

    def add_message(self, role: str, content: str, tokens: int = 0, time_taken: float = 0.0):
        """Add a message to history"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens
        })

        if role == "assistant":
            self.stats["total_tokens"] += tokens
            self.stats["total_time"] += time_taken
            self.stats["message_count"] += 1

    def clear(self):
        """Clear conversation history"""
        self.history = []
        self.stats["message_count"] = 0
        self.stats["total_tokens"] = 0
        self.stats["total_time"] = 0.0

    def save(self, filename: Optional[str] = None) -> str:
        """Save session to file"""
        if not filename:
            filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.session_dir / filename
        data = {
            "history": self.history,
            "stats": self.stats,
            "saved_at": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def load(self, filename: str) -> bool:
        """Load session from file"""
        filepath = self.session_dir / filename
        if not filepath.exists():
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.history = data.get("history", [])
        self.stats = data.get("stats", self.stats)
        return True

    def get_formatted_history(self) -> List[Dict[str, str]]:
        """Get history formatted for model input"""
        # Filter to only user and assistant messages
        return [msg for msg in self.history if msg["role"] in ["user", "assistant"]]

class InteractiveChat:
    """Main interactive chat interface"""

    def __init__(self, model_path: str = "models/latest", gpu_id: int = 1,
                 max_seq_length: int = 2048, reasoning_effort: str = "medium"):

        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Initialize console (Rich or fallback)
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

        # Chat parameters
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.reasoning_effort = reasoning_effort
        self.model = None
        self.tokenizer = None
        self.session = ChatSession()

        # Generation settings
        self.generation_config = {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "use_cache": True,
        }

    def print(self, content, style=None, panel=False):
        """Print with Rich formatting if available"""
        if self.console and RICH_AVAILABLE:
            if panel:
                self.console.print(Panel(content, style=style))
            else:
                self.console.print(content, style=style)
        else:
            print(content)

    def load_model(self):
        """Load the model using Unsloth approach"""
        self.print("\n🚀 Loading GPT-OSS-20B model...", style="bold yellow")
        start_time = time.time()

        try:
            # Always load base model first (Unsloth approach)
            self.print("Loading base model (4-bit quantized)...", style="dim")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            # Apply LoRA adapter if using fine-tuned model
            if not self.model_path.startswith("unsloth/"):
                self.print(f"Applying LoRA adapter from {self.model_path}...", style="dim")
                self.model = PeftModel.from_pretrained(self.model, self.model_path)

            # Enable inference mode (Unsloth optimization)
            FastLanguageModel.for_inference(self.model)

            load_time = time.time() - start_time
            self.print(f"✅ Model loaded successfully in {load_time:.1f} seconds!", style="bold green")
            self.print(f"   • Max sequence length: {self.max_seq_length} tokens", style="dim")
            self.print(f"   • VRAM usage: ~12.4GB", style="dim")

            return True

        except Exception as e:
            self.print(f"❌ Failed to load model: {str(e)}", style="bold red")
            return False

    def generate_streaming(self, prompt: str) -> Tuple[str, int, float]:
        """Generate response with real-time streaming"""

        # Format messages with chat template
        messages = []

        # Add system message
        messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant."
        })

        # Add conversation history (last 10 messages to keep context manageable)
        history = self.session.get_formatted_history()[-10:]
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": prompt
        })

        # Apply chat template (official Unsloth/HF approach)
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=self.reasoning_effort,  # GPT-OSS specific
        )

        # Tokenize
        inputs = self.tokenizer([formatted], return_tensors="pt").to("cuda")
        input_length = inputs.input_ids.shape[1]

        # Setup streaming
        if RICH_AVAILABLE:
            # Use TextIteratorStreamer for custom handling
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=False,  # Keep to handle template markers
            )

            # Start generation in thread
            generation_kwargs = {
                **inputs,
                **self.generation_config,
                "streamer": streamer,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream output with Rich formatting
            self.print("\n" + "─" * 60, style="dim")
            if self.console:
                self.console.print("Assistant: ", style="bold green", end="")
            else:
                print("Assistant: ", end="")

            generated_text = ""
            start_time = time.time()

            # Collect and display streaming tokens
            for new_text in streamer:
                # Clean up GPT-OSS specific markers
                if "<|channel|>" in new_text:
                    continue
                if "<|message|>" in new_text:
                    new_text = new_text.replace("<|message|>", "")
                if "<|return|>" in new_text or "<|end|>" in new_text:
                    break

                generated_text += new_text
                print(new_text, end="", flush=True)

            thread.join()
            elapsed = time.time() - start_time

        else:
            # Fallback: use basic TextStreamer
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)

            self.print("\nAssistant: ", end="")
            start_time = time.time()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config,
                    streamer=streamer,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            elapsed = time.time() - start_time

            # Decode for metrics
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )

        # Calculate metrics
        tokens_generated = len(self.tokenizer.encode(generated_text))
        tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

        # Display metrics
        self.print(f"\n\n[{tokens_generated} tokens in {elapsed:.1f}s = {tokens_per_sec:.1f} tokens/sec]",
                  style="dim cyan")
        self.print("─" * 60, style="dim")

        return generated_text.strip(), tokens_generated, elapsed

    def show_help(self):
        """Display help information"""
        if RICH_AVAILABLE:
            table = Table(title="Available Commands", show_header=True, header_style="bold magenta")
            table.add_column("Command", style="cyan", width=20)
            table.add_column("Description", style="white")

            commands = [
                ("/help", "Show this help message"),
                ("/clear", "Clear conversation history"),
                ("/save [filename]", "Save conversation to file"),
                ("/load <filename>", "Load conversation from file"),
                ("/stats", "Show session statistics"),
                ("/reasoning <low/medium/high>", "Set reasoning effort level"),
                ("/settings", "Show current settings"),
                ("/exit or /quit", "Exit the chat"),
                ("Ctrl+C", "Cancel current generation"),
            ]

            for cmd, desc in commands:
                table.add_row(cmd, desc)

            self.console.print(table)
        else:
            print("\nAvailable Commands:")
            print("  /help              - Show this help message")
            print("  /clear             - Clear conversation history")
            print("  /save [filename]   - Save conversation to file")
            print("  /load <filename>   - Load conversation from file")
            print("  /stats             - Show session statistics")
            print("  /reasoning <level> - Set reasoning effort (low/medium/high)")
            print("  /settings          - Show current settings")
            print("  /exit or /quit     - Exit the chat")
            print("  Ctrl+C             - Cancel current generation")

    def show_stats(self):
        """Display session statistics"""
        stats = self.session.stats
        avg_speed = (stats["total_tokens"] / stats["total_time"]) if stats["total_time"] > 0 else 0

        if RICH_AVAILABLE:
            table = Table(title="Session Statistics", show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")

            table.add_row("Messages", str(stats["message_count"]))
            table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
            table.add_row("Total Time", f"{stats['total_time']:.1f} seconds")
            table.add_row("Average Speed", f"{avg_speed:.1f} tokens/sec")
            table.add_row("Session Duration", str(datetime.now() - stats["session_start"]).split('.')[0])

            self.console.print(table)
        else:
            print("\nSession Statistics:")
            print(f"  Messages: {stats['message_count']}")
            print(f"  Total Tokens: {stats['total_tokens']:,}")
            print(f"  Total Time: {stats['total_time']:.1f} seconds")
            print(f"  Average Speed: {avg_speed:.1f} tokens/sec")
            print(f"  Session Duration: {datetime.now() - stats['session_start']}")

    def show_settings(self):
        """Display current settings"""
        if RICH_AVAILABLE:
            table = Table(title="Current Settings", show_header=False)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="yellow")

            table.add_row("Model Path", self.model_path)
            table.add_row("Max Sequence Length", str(self.max_seq_length))
            table.add_row("Reasoning Effort", self.reasoning_effort)
            table.add_row("Temperature", str(self.generation_config["temperature"]))
            table.add_row("Max New Tokens", str(self.generation_config["max_new_tokens"]))
            table.add_row("Top P", str(self.generation_config["top_p"]))
            table.add_row("Top K", str(self.generation_config["top_k"]))

            self.console.print(table)
        else:
            print("\nCurrent Settings:")
            print(f"  Model Path: {self.model_path}")
            print(f"  Max Sequence Length: {self.max_seq_length}")
            print(f"  Reasoning Effort: {self.reasoning_effort}")
            print(f"  Temperature: {self.generation_config['temperature']}")
            print(f"  Max New Tokens: {self.generation_config['max_new_tokens']}")

    def run(self):
        """Main chat loop"""

        # Welcome message
        if RICH_AVAILABLE:
            welcome = Panel.fit(
                "[bold cyan]GPT-OSS-20B Interactive Chat[/bold cyan]\n"
                "[dim]Powered by Unsloth • 15-16 tokens/sec on RTX 3090[/dim]\n\n"
                "[yellow]Type /help for commands • Ctrl+C to cancel generation[/yellow]",
                border_style="bright_blue"
            )
            self.console.print(welcome)
        else:
            print("\n" + "="*60)
            print("GPT-OSS-20B Interactive Chat")
            print("Powered by Unsloth • 15-16 tokens/sec on RTX 3090")
            print("="*60)
            print("\nType /help for commands • Ctrl+C to cancel generation\n")

        # Load model
        if not self.load_model():
            return

        # Main loop
        while True:
            try:
                # Get user input
                if RICH_AVAILABLE:
                    user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                else:
                    user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command_parts = user_input.split()
                    command = command_parts[0].lower()

                    if command in ["/exit", "/quit"]:
                        self.print("\n👋 Goodbye!", style="bold yellow")
                        break

                    elif command == "/help":
                        self.show_help()

                    elif command == "/clear":
                        self.session.clear()
                        self.print("✅ Conversation history cleared", style="green")

                    elif command == "/save":
                        filename = command_parts[1] if len(command_parts) > 1 else None
                        saved_path = self.session.save(filename)
                        self.print(f"✅ Conversation saved to {saved_path}", style="green")

                    elif command == "/load":
                        if len(command_parts) < 2:
                            self.print("❌ Please specify a filename", style="red")
                        else:
                            if self.session.load(command_parts[1]):
                                self.print(f"✅ Conversation loaded from {command_parts[1]}", style="green")
                            else:
                                self.print(f"❌ Could not load {command_parts[1]}", style="red")

                    elif command == "/stats":
                        self.show_stats()

                    elif command == "/settings":
                        self.show_settings()

                    elif command == "/reasoning":
                        if len(command_parts) < 2:
                            self.print(f"Current reasoning effort: {self.reasoning_effort}", style="yellow")
                        else:
                            level = command_parts[1].lower()
                            if level in ["low", "medium", "high"]:
                                self.reasoning_effort = level
                                self.print(f"✅ Reasoning effort set to: {level}", style="green")
                            else:
                                self.print("❌ Invalid level. Use: low, medium, or high", style="red")

                    else:
                        self.print(f"❌ Unknown command: {command}", style="red")
                        self.print("Type /help for available commands", style="dim")

                    continue

                # Add user message to history
                self.session.add_message("user", user_input)

                # Generate response with streaming
                try:
                    response, tokens, elapsed = self.generate_streaming(user_input)

                    # Add assistant response to history
                    self.session.add_message("assistant", response, tokens, elapsed)

                except KeyboardInterrupt:
                    self.print("\n\n⚠️  Generation cancelled", style="yellow")
                    continue
                except Exception as e:
                    self.print(f"\n❌ Error generating response: {str(e)}", style="red")
                    continue

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                self.print("\n\n💡 Tip: Use /exit to quit", style="dim yellow")
                continue
            except EOFError:
                # Handle Ctrl+D
                self.print("\n\n👋 Goodbye!", style="bold yellow")
                break
            except Exception as e:
                self.print(f"\n❌ Unexpected error: {str(e)}", style="red")
                continue

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Interactive chat with GPT-OSS-20B")
    parser.add_argument("--model_path", type=str, default="models/latest",
                       help="Path to model or 'models/latest' for most recent")
    parser.add_argument("--gpu", type=int, default=1,
                       help="GPU to use (0 or 1)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--reasoning", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="Initial reasoning effort level")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature (0.0-1.0)")
    parser.add_argument("--max_tokens", type=int, default=500,
                       help="Maximum tokens to generate per response")

    args = parser.parse_args()

    # Create and run chat interface
    chat = InteractiveChat(
        model_path=args.model_path,
        gpu_id=args.gpu,
        max_seq_length=args.max_seq_length,
        reasoning_effort=args.reasoning
    )

    # Update generation config from args
    chat.generation_config["temperature"] = args.temperature
    chat.generation_config["max_new_tokens"] = args.max_tokens

    # Run the chat
    try:
        chat.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()