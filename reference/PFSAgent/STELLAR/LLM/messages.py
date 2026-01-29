


    
class MessageThread:
    def __init__(self, messages: list[dict]):
        self.messages = messages

    def __str__(self):
        return "\n".join([str(message) for message in self.messages])

    def __iter__(self):
        """Make the MessageThread iterable, yielding each message."""
        return iter(self.messages)

    def __getitem__(self, index):
        """Make the MessageThread subscriptable, allowing access via indexing."""
        return self.messages[index]

    def add_messages(self, messages: list[dict]):
        self.messages.extend(messages)

    def get_messages(self):
        return self.messages
    
    def dump(self):
        return [{"role": message["role"], "content": message["content"]} for message in self.messages]
    
    
    
    