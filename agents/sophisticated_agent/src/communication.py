"""
FIPA-ACL Communication Protocol Implementation
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from pydantic import BaseModel, Field


class Performative(str, Enum):
    """FIPA-ACL Performatives"""

    # Requesting information or action
    REQUEST = "request"
    QUERY = "query"

    # Providing information
    INFORM = "inform"
    CONFIRM = "confirm"

    # Negotiation
    PROPOSE = "propose"
    ACCEPT_PROPOSAL = "accept-proposal"
    REJECT_PROPOSAL = "reject-proposal"

    # Error handling
    FAILURE = "failure"
    NOT_UNDERSTOOD = "not-understood"

    # Task delegation
    CFP = "cfp"  # Call for proposal
    REFUSE = "refuse"
    AGREE = "agree"


class FIPAMessage(BaseModel):
    """FIPA-ACL Message Format"""

    # Required fields
    performative: Performative
    sender: str
    receiver: str
    content: Dict[str, Any]

    # Optional fields
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reply_with: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    in_reply_to: Optional[str] = None
    language: str = "json"
    ontology: str = "bioinformatics-v1"
    protocol: str = "fipa-request"
    reply_by: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "performative": "request",
                "sender": "agent-123",
                "receiver": "agent-456",
                "content": {"action": "analyze_sequence", "sequence": "ATCGATCG"},
                "conversation_id": "conv-789",
                "reply_with": "msg-001",
            }
        }


class ConversationManager:
    """Manages FIPA-ACL conversations"""

    def __init__(self):
        self.conversations: Dict[str, List[FIPAMessage]] = {}

    def add_message(self, message: FIPAMessage):
        """Add message to conversation"""
        conv_id = message.conversation_id
        if conv_id not in self.conversations:
            self.conversations[conv_id] = []
        self.conversations[conv_id].append(message)

    def get_conversation(self, conversation_id: str) -> List[FIPAMessage]:
        """Get all messages in a conversation"""
        return self.conversations.get(conversation_id, [])

    def get_last_message(self, conversation_id: str) -> Optional[FIPAMessage]:
        """Get the last message in a conversation"""
        conv = self.get_conversation(conversation_id)
        return conv[-1] if conv else None

    def create_reply(
        self,
        original_message: FIPAMessage,
        performative: Performative,
        content: Dict[str, Any],
        sender: str,
    ) -> FIPAMessage:
        """Create a reply to a message"""

        return FIPAMessage(
            performative=performative,
            sender=sender,
            receiver=original_message.sender,
            content=content,
            conversation_id=original_message.conversation_id,
            in_reply_to=original_message.reply_with,
            reply_with=str(uuid.uuid4()),
        )


class MessageValidator:
    """Validates FIPA-ACL messages"""

    @staticmethod
    def validate_request(message: FIPAMessage) -> bool:
        """Validate REQUEST message"""
        content = message.content
        return "action" in content

    @staticmethod
    def validate_query(message: FIPAMessage) -> bool:
        """Validate QUERY message"""
        content = message.content
        return "query_type" in content

    @staticmethod
    def validate_inform(message: FIPAMessage) -> bool:
        """Validate INFORM message"""
        return len(message.content) > 0

    @staticmethod
    def validate_message(message: FIPAMessage) -> bool:
        """Validate any message"""
        validators = {
            Performative.REQUEST: MessageValidator.validate_request,
            Performative.QUERY: MessageValidator.validate_query,
            Performative.INFORM: MessageValidator.validate_inform,
        }

        validator = validators.get(message.performative, lambda m: True)
        return validator(message)
