import os

class ApiKeyManager:
    """
    Manages a list of API keys to rotate through when quota limits are hit.
    This version automatically discovers numbered API keys from environment variables.
    """
    def __init__(self):
        self.keys = []
        i = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                self.keys.append(key)
                i += 1
            else:
                break
        
        if not self.keys:
            raise ValueError("No API keys found. Please set GEMINI_API_KEY_1, etc., in your .env file.")
        
        self.current_key_index = 0
        print(f"🔑 ApiKeyManager initialized with {len(self.keys)} numbered Gemini keys.")

    def get_current_key(self):
        """Returns the current API key."""
        return self.keys[self.current_key_index]

    def get_next_key(self):
        """Rotates to the next key and returns it."""
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        print(f"🔄 Switching to API key index: {self.current_key_index}")
        return self.get_current_key()