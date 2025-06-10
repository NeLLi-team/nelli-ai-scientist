"""
Tests for BioinformaticsAgent
"""

import pytest
from src.agent import BioinformaticsAgent, AgentConfig
from src.communication import FIPAMessage, Performative
from src.llm_interface import LLMProvider


@pytest.fixture
def agent_config():
    """Create test agent configuration"""
    return AgentConfig(
        name="test-agent",
        role="test_analyst",
        capabilities=["test_capability"],
        llm_provider=LLMProvider.CBORG,
    )


@pytest.fixture
def agent(agent_config):
    """Create test agent"""
    return BioinformaticsAgent(agent_config)


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initializes correctly"""
    assert agent.config.name == "test-agent"
    assert agent.config.role == "test_analyst"
    assert "sequence_stats" in agent.tools.list_tools()


@pytest.mark.asyncio
async def test_capability_query(agent):
    """Test agent responds to capability queries"""
    message = FIPAMessage(
        performative=Performative.QUERY,
        sender="test-user",
        receiver=agent.agent_id,
        content={"query_type": "capabilities"},
    )

    response = await agent.process_message(message)

    assert response.performative == Performative.INFORM
    assert "capabilities" in response.content
    assert "tools" in response.content


@pytest.mark.asyncio
async def test_sequence_analysis_request(agent):
    """Test sequence analysis request"""
    message = FIPAMessage(
        performative=Performative.REQUEST,
        sender="test-user",
        receiver=agent.agent_id,
        content={
            "action": "analyze",
            "data": "ATCGATCGATCG",
            "analysis_type": "basic_stats",
        },
    )

    response = await agent.process_message(message)

    assert response.performative == Performative.INFORM
    assert response.content["status"] == "success"


@pytest.mark.asyncio
async def test_not_understood_response(agent):
    """Test agent returns NOT_UNDERSTOOD for invalid requests"""
    message = FIPAMessage(
        performative=Performative.REQUEST,
        sender="test-user",
        receiver=agent.agent_id,
        content={"unknown": "request"},
    )

    response = await agent.process_message(message)

    assert response.performative == Performative.NOT_UNDERSTOOD


@pytest.mark.asyncio
async def test_conversation_history(agent):
    """Test conversation history is maintained"""
    message = FIPAMessage(
        performative=Performative.INFORM,
        sender="test-user",
        receiver=agent.agent_id,
        content={"info": "test information"},
    )

    await agent.process_message(message)

    assert len(agent.conversation_history) == 1
    assert (
        agent.conversation_history[0]["incoming"]["content"]["info"]
        == "test information"
    )
