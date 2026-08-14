import json
import logging
from typing import Annotated

from crewai import TaskOutput
from fastapi import APIRouter, Body, Form, Header, HTTPException, UploadFile

from app.core.config import settings
from app.helpers.guardrails import GaurdRails
from app.models.AgentRequest import AgentRequest
from app.models.ConfigSingleAgentModel import ConfigBase
from app.models.healthResponse import HealthResponse
from app.models.ResponseModel import ResponseModel
from app.models.test_tool import TestTool
from app.services.agent_ai import AgentQueryExecutor
from app.services.agent_ai import execute as execute_agent
from app.services.agent_ai import execute_files as execute_agent_files
from app.services.health_check import health_check
from app.services.tool_test import extract_params

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="🏥 Health Check",
    description="""
    **Comprehensive health check endpoint** that provides deailted information about:
    """,
    response_description="Detailed health check information",
    tags=["Health & Monitoring"])

async def health_check_endpoint():
    try:
        return await health_check()
    except Exception as e:
        logger.error(f"Unexpected error in health check: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error during health check: {e}"
        )


@router.post("/config-refresh",summary="🔄 Refresh Configuration",
    description="""
    **Reload configuration from remote config service.**
    
    This endpoint triggers a refresh of application configuration from the configured
    remote config service (Azure App Configuration, etc.).
    """,
    tags=["Configuration"])

async def config_refresh_endpoint():
    try:
        is_loaded, message = settings.load_remote_config()
        assert is_loaded == True
        return {"message": message}
    except Exception as e:
        logger.error(f"Unexpected error in Config Refresh: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during Config Refresh:: {message}",
        )


@router.post(
    "/test_tool", summary="🧪 Test Tool Pipeline",
    description="""
    **Execute tool testing pipeline** through both Pipeline and Agent services.
    
    This endpoint processes the input payload through:
    **Agent Service** - Executes the tool with the agent framework
    
    **Security:** Requires valid JWT Bearer token in Authorization header.""",
    tags=["Tool Testing"]
)
async def test_tool_endpoint(payload: TestTool, Authorization: str = Header(None)):
    try:
        return await extract_params(payload, Authorization)
    except Exception as e:
        logger.error(f"Error in test tool endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while testing the tool code: {e}",
        )


@router.post(
    "/run",
    summary="🤖 Execute Agent",
    description="""
    **Execute an AI agent** through the Agent Service.
    
    This endpoint processes requests through the configured AI agent framework:
    * Validates agent configuration and parameters
    * Executes agent with tools and context
    * Manages conversation state and memory
    * Returns agent response and execution metadata""",
    tags=["Agent Execution"]
)
async def execute_agent_endpoint(
    Authorization: Annotated[str | None, Header()] = None,
    agentRequest: AgentRequest = Body(...),
):
    try:
        return await execute_agent(Authorization, agentRequest)
    except Exception as e:
        logger.error(f"Error in agent execution endpoint: {e}")
        if e.status_code == 403:
            raise HTTPException(
                status_code=403, detail="Forbidden: You don't have permission to access this resource. Please check tool consent."
            )
        else:
            raise HTTPException(
            status_code=500, detail=f"An error occurred during agent execution: {e}"
        )


@router.post(
    "/run-files",
    summary="🤖📁 Execute Agent with Files",
    description="""
    **Execute AI agent with file attachments** for document-aware interactions.
    
    Enables agents to process and reason about files:
    * Document analysis and Q&A
    * Image understanding and description
    * Code review and analysis
    * Multi-file context reasoning""",
    tags=["Agent Execution"]
)
async def execute_agent_files_endpoint(
    Authorization: Annotated[str | None, Header()] = None,
    files: list[UploadFile] = None,
    agentId: Annotated[str, Form()] = None,
    userInputs: Annotated[str, Form()] = None,  # JSON string
    user: Annotated[str, Form()] = None,
    executionId: Annotated[str, Form()] = None,
    tools: Annotated[str | None, Form()] = "[]",  # JSON string
    userTools: Annotated[str | None, Form()] = "[]",  # JSON string
):
    try:
        if not executionId.strip():
            logger.error("Execution ID is required")
            raise HTTPException(status_code=400, detail="Execution ID is required")
        request_data = AgentRequest(
            agentId=int(agentId),
            executionId=executionId,
            userInputs=json.loads(userInputs),
            user=user,
            tools=json.loads(tools),
            userTools=json.loads(userTools),
        )
        return await execute_agent_files(Authorization, files, request_data)
    except Exception as e:
        logger.error(f"Error in agent execution endpoint: {e}")
        if e.status_code == 403:
            raise HTTPException(
                status_code=403, detail="Forbidden: You don't have permission to access this resource. Please check tool consent."
            )
        else:
            raise HTTPException(
            status_code=500, detail=f"An error occurred during agent execution: {e}"
        )


@router.post("/query", summary="💬 Query Agent",
    description="""
    **Send a query to the agent** for conversational interactions.
    
    This endpoint provides a flexible interface for agent queries:
    * Accepts raw JSON payload for maximum flexibility
    * Supports various query formats and structures
    * Manages conversation context automatically
    * Returns structured agent responses)""",
    tags=["Agent Execution"]
    )
async def route_request(
    request: ConfigBase, Authorization: Annotated[str | None, Header()] = None
):
    try:
        # guardrail_validator
        validator = None
        if request.nemo_guardrails:
            validator = GaurdRails(
                Authorization=Authorization, request_payload=request.model_dump()
            ).validation_closure()

            guardrails_response = validator(request.prompt)

             # if guardrails block input prompt
            if not guardrails_response[0]:
                return ResponseModel(
                    input=request.prompt,
                    answer=str(guardrails_response[1]) if guardrails_response[1] else "No response generated.",
                    context=[],
                )

        agent_executor = AgentQueryExecutor(
            request=request,
            Authorization=Authorization,
            validator=validator,
        )
        result = await agent_executor.execute_query()

        return ResponseModel(
            input=request.prompt,
            answer=str(result) if result else "No response generated.",
            context=[],
        )

    except Exception as e:
        logger.error(f"Unexpected error in route_request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
