# Agent System Prompts

This directory contains the system prompts used by the Universal MCP Agent. These prompts are managed by the `PromptManager` class and can be easily modified without changing the agent code.

## Available Prompts

### `tool_selection.txt`
Used when processing natural language input to determine which tools to use or whether to provide a direct answer.

**Variables:**
- `{tools_context}` - Formatted list of available tools
- `{user_input}` - The user's request

**Output:** JSON response with intent, response type, suggested tools, or clarification needs.

### `reflection.txt`
Used after tool execution to analyze and interpret the results.

**Variables:**
- `{user_request}` - The original user request
- `{results_summary}` - Summary of tool execution results

**Output:** Conversational analysis and interpretation of the tool results.

### `general_response.txt`
Used for generating responses when no tools are needed (general knowledge questions).

**Variables:**
- `{user_input}` - The user's request

**Output:** Direct conversational response.

## Usage

The `PromptManager` class automatically loads and formats these prompts:

```python
# Load and format a prompt
prompt = self.prompt_manager.format_prompt(
    "tool_selection",
    tools_context=tools_context,
    user_input=user_input
)
```

## Modifying Prompts

You can modify any prompt file to customize the agent's behavior:

1. Edit the `.txt` file directly
2. The changes will be loaded automatically (prompts are cached but can be reloaded)
3. Use `agent.prompt_manager.reload_prompts()` to force reload during development

## Adding New Prompts

1. Create a new `.txt` file in this directory
2. Use `{variable_name}` placeholders for dynamic content
3. Use `prompt_manager.format_prompt("new_prompt_name", **variables)` in the agent code

## Best Practices

- Keep prompts focused and specific to their purpose
- Use clear variable names with descriptive placeholders
- Test prompts with various inputs to ensure robust behavior
- Document any special formatting requirements or expected outputs