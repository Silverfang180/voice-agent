"""AI Receptionist Brain using Groq LLM for ABC Business Services."""
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Attempt loading dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "qwen/qwen3.8-27b"


class ReceptionistBrain:
    def __init__(self, model_name: Optional[str] = None, faq_path: str = "business_faq.json"):
        # Load API key
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. Please add it to your .env file or environment."
            )

        # Configure model
        self.model_name = os.environ.get("GROQ_LLM_MODEL", model_name or DEFAULT_MODEL).strip()

        # Import and initialize Groq client
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("The 'groq' package is not installed. Install it with: pip install groq")

        # Load business FAQ
        self.faq_path = os.path.abspath(faq_path)
        if not os.path.isfile(self.faq_path):
            raise FileNotFoundError(f"FAQ file not found at '{self.faq_path}'")
        
        with open(self.faq_path, "r", encoding="utf-8") as f:
            self.faq_data = json.load(f)

        # Conversation history stored per session_id: {session_id: [{"role": ..., "content": ...}]}
        self.sessions: Dict[str, List[dict]] = {}

        # Cache of verified models to avoid redundant API queries
        self._verified_models = set()

    def verify_model_available(self) -> bool:
        """Verify whether the selected model is supported and accessible in the Groq account."""
        if self.model_name in self._verified_models:
            return True
        try:
            models_resp = self.client.models.list()
            available_ids = [m.id for m in models_resp.data]
            if self.model_name in available_ids:
                self._verified_models.add(self.model_name)
                return True
            # If not in the list, test with a 1-token call
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1
            )
            self._verified_models.add(self.model_name)
            return True
        except Exception as e:
            print(f"Warning: Model verification failed for '{self.model_name}': {e}", file=sys.stderr)
            return False

    def get_system_prompt(self, language: str = "hi") -> str:
        """Construct a natural, conversational receptionist prompt dynamically from business_faq.json."""
        # Pass all factual business fields directly as compact JSON (excluding meta notices/legacy scripts)
        facts = {k: v for k, v in self.faq_data.items() if k not in ("_notice", "faq")}
        business_name = (
            facts.get("business_name")
            or facts.get("company_profile", {}).get("name")
            or "ABC Business Services"
        )
        facts_json = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))

        if language == "hi":
            return (
                f"आप {business_name} के विनम्र, स्वाभाविक फोन रिसेप्शनिस्ट हैं और हिंदी व मराठी दोनों भाषाओं में बात कर सकते हैं। "
                "ग्राहक का इरादा और पिछली बातचीत समझकर नीचे दिए गए JSON बिज़नेस डेटा के आधार पर "
                "अपने शब्दों में 1-2 छोटे, स्पष्ट और स्वाभाविक हिंदी वाक्यों में जवाब दें (कोई बुलेट या मार्कडाउन नहीं)। "
                "अभिवादन का विनम्र उत्तर दें और जहाँ उपयोगी हो एक छोटा अनुवर्ती सवाल पूछें। "
                "डेटा में न दी गई कोई भी कीमत, समय, पता या सेवा अपनी तरफ से न गढ़ें; जानकारी न होने पर विनम्रता से मना करें।\n\n"
                f"Business Data (JSON):\n{facts_json}"
            )
        else:
            return (
                f"तुम्ही {business_name} चे नम्र, नैसर्गिक फोन रिसेप्शनिस्ट आहात आणि हिंदी व मराठी दोन्ही भाषांमध्ये बोलू शकता. "
                "ग्राहकाचा हेतू आणि मागील संवाद समजून खालील JSON बिझनेस डेटाच्या आधारे "
                "स्वतःच्या शब्दांत 1-2 लहान, स्पष्ट आणि नैसर्गिक मराठी वाक्यांत उत्तर द्या (बुलेट्स किंवा मार्कडाउन नको). "
                "अभिवादनाला नम्र उत्तर द्या आणि योग्य तिथे एक छोटा पुढचा प्रश्न विचारा. "
                "डेटामध्ये नसलेली कोणतीही किंमत, वेळ, पत्ता किंवा सेवा स्वतःहून तयार करू नका; माहिती नसल्यास नम्रपणे तसे सांगा.\n\n"
                f"Business Data (JSON):\n{facts_json}"
            )

    def ask(self, user_question: str, session_id: str = "default", language: str = "hi") -> dict:
        """Submit a user query to the receptionist, maintaining session context."""
        # Ensure session history exists
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        system_prompt = self.get_system_prompt(language=language)

        # Build messages payload with compact system prompt and sliding history window (last 6 messages)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.sessions[session_id][-6:])
        messages.append({"role": "user", "content": user_question})

        start_time = time.perf_counter()
        try:
            from groq import AuthenticationError, RateLimitError, APIConnectionError, APIStatusError
        except ImportError:
            AuthenticationError = RateLimitError = APIConnectionError = APIStatusError = Exception

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                reasoning_effort="none",
                temperature=0.35,  # Natural variation while staying strictly factual
                max_tokens=130,    # Prevents mid-sentence truncation in Devanagari
            )
            response_time = time.perf_counter() - start_time
            reply = response.choices[0].message.content.strip()
        except AuthenticationError:
            return {
                "error": "AuthenticationError: Invalid GROQ_API_KEY. Please verify your credentials.",
                "response_time": time.perf_counter() - start_time,
                "session_id": session_id,
            }
        except RateLimitError as e:
            return {
                "error": f"RateLimitError: Groq rate limit exceeded: {e}",
                "response_time": time.perf_counter() - start_time,
                "session_id": session_id,
            }
        except (APIConnectionError, APIStatusError) as e:
            return {
                "error": f"GroqAPIError: {e}",
                "response_time": time.perf_counter() - start_time,
                "session_id": session_id,
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {e}",
                "response_time": time.perf_counter() - start_time,
                "session_id": session_id,
            }

        # Update session memory
        self.sessions[session_id].append({"role": "user", "content": user_question})
        self.sessions[session_id].append({"role": "assistant", "content": reply})

        return {
            "reply": reply,
            "response_time": response_time,
            "model": self.model_name,
            "session_id": session_id,
            "language": language,
            "turn_count": len(self.sessions[session_id]) // 2,
        }

    def reset_session(self, session_id: str = "default"):
        """Clear memory for a given session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


if __name__ == "__main__":
    brain = ReceptionistBrain()
    print("Receptionist Brain initialized.")
    print("Selected Model:", brain.model_name)
    print("Model Available:", brain.verify_model_available())

