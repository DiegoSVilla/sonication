"""Pizza Agent — Simplified Example

Demonstrates the Sonication SDK with LLM context management.

Usage:
    pip install sonication
    python examples/pizza_agent.py

The agent will prompt you to type messages. Context is managed
automatically by LLMNode — no manual history tracking needed.
"""
import asyncio
import logging

import sonication

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PIZZA_SYSTEM_PROMPT = """You are Marco, the friendly pizza ordering assistant.

You work at Marco's Pizza and help customers order pizzas over a phone call.

Rules:
- Ask for the pizza size (small, medium, large)
- Ask about toppings (pepperoni, mushrooms, olives, onions, extra cheese)
- Ask if they want drinks or sides
- Confirm the order before placing it
- Be warm, concise, and speak in short sentences
- Never use lists, markdown, or emojis
- Always confirm the total price
"""


class PizzaAgent:
    """Pizza ordering agent with automatic context management."""

    def __init__(
        self,
        stt_url: str = "http://localhost:8092",
        llm_url: str = "http://localhost:8093",
        tts_url: str = "http://localhost:8094",
    ):
        # Create nodes
        self.stt_node = sonication.STTNode(stt_url)
        self.llm_node = sonication.LLMNode(
            llm_url,
            system_prompt=PIZZA_SYSTEM_PROMPT,
        )
        self.tts_node = sonication.TTSNode(tts_url, voice="Marco", language="English")

        # Create pipeline
        self.pipeline = sonication.HotPipe(
            pipeline_type=sonication.PipelineType.SI_SO_THREE_STEP_PIPELINE_CHAT
        )
        self.pipeline.add_node(self.stt_node)
        self.pipeline.add_node(self.llm_node)
        self.pipeline.add_node(self.tts_node)
        self.pipeline.connect()

        self.turn_count = 0

    async def warmup(self):
        """Warm up all node connections."""
        await self.stt_node.warmup()
        await self.llm_node.warmup()
        await self.tts_node.warmup()

    async def process_text(self, text: str) -> dict:
        """Process text input through the full pipeline."""
        self.turn_count += 1
        logger.info(f"Turn {self.turn_count}: {text[:50]}...")

        # Run the pipeline
        result = await self.pipeline.turn("stt", text)

        return {
            "turn_index": self.turn_count,
            "stt_text": result.get("stt_text", ""),
            "llm_response": result.get("llm_response", ""),
            "shot_latency_ms": result.get("shot_latency_ms", 0),
        }

    def get_llm_history(self) -> list[dict]:
        """Inspect the LLMNode's internal conversation history."""
        return self.llm_node.get_history()


async def main():
    """Run the pizza agent with text input."""
    agent = PizzaAgent()
    await agent.warmup()
    logger.info("Pizza agent ready!")
    logger.info("Type messages to chat with Marco. Enter 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "quit":
                break
            if not user_input:
                continue

            result = await agent.process_text(user_input)
            print(f"\nMarco: {result['llm_response']}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")

    logger.info("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())