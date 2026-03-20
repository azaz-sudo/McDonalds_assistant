import os
import json
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import httpx

load_dotenv()

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def call_mcp_tool(tool_name: str, arguments: dict = None, config: dict = None):
    if config is None:
        config = load_config()
    
    mcp_config = config.get('mcpServers', {}).get('mcd-mcp', {})
    mcp_url = mcp_config.get('url', 'https://mcp.mcd.cn')
    auth_token = mcp_config.get('headers', {}).get('Authorization', '')
    
    if arguments is None:
        arguments = {}
    
    async with httpx.AsyncClient(headers={"Authorization": auth_token}) as client:
        response = await client.post(
            mcp_url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
        )
        
        return response.json()

async def get_mcp_tools(config: dict = None):
    if config is None:
        config = load_config()
    
    mcp_config = config.get('mcpServers', {}).get('mcd-mcp', {})
    mcp_url = mcp_config.get('url', 'https://mcp.mcd.cn')
    auth_token = mcp_config.get('headers', {}).get('Authorization', '')
    
    async with httpx.AsyncClient(headers={"Authorization": auth_token}) as client:
        response = await client.post(
            mcp_url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {}
            }
        )
        
        result = response.json()
        return result.get('result', {}).get('tools', [])

async def extract_json_from_content(content):
    if isinstance(content, list):
        for item in content:
            if item.get('type') == 'text':
                text = item.get('text', '')
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
    elif isinstance(content, str):
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
    return None

async def process_user_query(user_message: str, config: dict):
    local_config = config.get('local', {})
    
    llm = ChatOpenAI(
        model_name=local_config.get('model', 'default-model'),
        api_key=local_config.get('api_key', 'dummy-key'),
        base_url=local_config.get('api_url', 'http://localhost:8080/v1'),
        temperature=0.7
    )
    
    tools = await get_mcp_tools(config)
    tools_description = json.dumps(tools, ensure_ascii=False, indent=2)
    
    system_prompt = f"""你是一个麦当劳助手，可以使用以下 MCP 工具：

{tools_description}

请根据用户的查询，自主决定使用哪个工具，并返回工具名称和参数。返回格式为 JSON：
{{
    "tool_name": "工具名称",
    "arguments": {{}}
}}

如果需要查询优惠券，使用 query-my-coupons 工具。
如果需要查询订单，使用 query-order 工具。
如果需要查询餐品，使用 query-meals 工具。
如果需要查询地址，使用 delivery-query-addresses 工具。
如果需要计算价格，使用 calculate-price 工具。
如果需要创建订单，使用 create-order 工具。
如果需要创建配送地址，使用 delivery-create-address 工具。

请直接返回 JSON 格式，不要返回其他内容。"""
    
    messages = [
        HumanMessage(content=f"{system_prompt}\n\n用户的问题是：{user_message}")
    ]
    
    response = llm.invoke(messages)
    
    decision = await extract_json_from_content(response.content)
    
    if decision is None:
        return {
            "success": False,
            "error": "无法从 AI 响应中提取 JSON 决策",
            "raw_response": str(response.content)
        }
    
    tool_name = decision.get("tool_name")
    arguments = decision.get("arguments", {})
    
    tool_result = await call_mcp_tool(tool_name, arguments, config)
    
    messages.extend([
        response,
        HumanMessage(content=f"工具 {tool_name} 的返回结果是：{json.dumps(tool_result, ensure_ascii=False)}")
    ])
    
    final_response = llm.invoke(messages)
    
    return {
        "success": True,
        "decision": decision,
        "tool_result": tool_result,
        "response": str(final_response.content)
    }

if __name__ == "__main__":
    config = load_config()
    user_query = "我现在有哪些优惠券"
    result = asyncio.run(process_user_query(user_query, config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
