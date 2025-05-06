import os
import json
from typing import List, Dict, Any
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from openai import OpenAI
from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.tools import get_current_weather


def load_reference_data() -> Dict[str, Any]:
    """Loads product data from data.json."""
    try:
        with open("api/data.json", "r") as f:
            data = json.load(f)
            # Convert list to dict keyed by 'reference' for faster lookup
            return {item["reference"]: item for item in data}
    except FileNotFoundError:
        print("Warning: api/data.json not found.")
        return {}
    except json.JSONDecodeError:
        print("Warning: api/data.json is not valid JSON.")
        return {}


reference_data_store = load_reference_data()


def load_company_margins_data() -> Dict[str, Any]:
    """Loads company margin data from company_margins.json."""
    try:
        with open("api/company_margins.json", "r") as f:
            data = json.load(f)
            return data  # Assuming the JSON is already structured as a dict keyed by company_id
    except FileNotFoundError:
        print("Warning: api/company_margins.json not found.")
        return {}
    except json.JSONDecodeError:
        print("Warning: api/company_margins.json is not valid JSON.")
        return {}


company_margins_store = load_company_margins_data()


def load_stock_data() -> Dict[str, Any]:
    """Loads stock data from stock.json."""
    try:
        with open("api/stock.json", "r") as f:
            data = json.load(f)
            return data  # Assuming JSON is keyed by product reference
    except FileNotFoundError:
        print("Warning: api/stock.json not found.")
        return {}
    except json.JSONDecodeError:
        print("Warning: api/stock.json is not valid JSON.")
        return {}


stock_data_store = load_stock_data()


def get_reference_data(product_reference: str) -> Dict[str, Any]:
    """Gets reference data for a given product reference ID."""
    product_info = reference_data_store.get(product_reference)
    if product_info:
        return product_info
    else:
        return {"error": f"Product with reference {product_reference} not found."}


def get_margins(company_id: str) -> dict:
    """Gets mock margin data for a given company ID."""
    company_info = company_margins_store.get(company_id)
    if company_info:
        return company_info
    else:
        return {"error": f"Company with ID {company_id} not found."}


def get_customer_id_for_company(company_id: str) -> dict:
    """Gets a mock customer ID for a given company ID from the company data store."""
    company_info = company_margins_store.get(company_id)
    if company_info and "customer_id" in company_info:
        return {"customer_id": company_info["customer_id"]}
    elif company_info:
        return {"error": f"Customer ID not found for company ID {company_id}."}
    else:
        return {"error": f"Company with ID {company_id} not found."}


def get_product_stock_info(product_reference: str) -> dict:
    """Gets stock information for a given product reference ID."""
    stock_info = stock_data_store.get(product_reference)
    if stock_info:
        # Ensure all expected keys are present before formatting
        name = stock_info.get("name", "N/A")
        stock = stock_info.get("stock", "N/A")
        restock_date = stock_info.get("restock_date", "N/A")
        restocking_number = stock_info.get("restocking_number", "N/A")

        return {
            "message": f"For product {name} (Ref: {product_reference}), we have {stock} in stock, with {restocking_number} new units arriving around {restock_date}."
        }
    else:
        return {
            "error": f"Stock information for product reference {product_reference} not found."
        }


def get_product_price_info(product_reference: str, company_id: str) -> dict:
    """
    Calculates and returns selling price information for a product,
    considering the company-specific margin. If the company ID is unknown,
    the assistant should ask the user for it.
    """
    product_data = reference_data_store.get(product_reference)
    if not product_data:
        return {"error": f"Product with reference {product_reference} not found."}

    base_cost_str = product_data.get("base_cost")
    if base_cost_str is None:  # Check if base_cost key exists
        return {"error": f"Base cost for product {product_reference} not found."}
    try:
        base_cost = float(base_cost_str)
    except ValueError:
        return {
            "error": f"Invalid base cost format for product {product_reference}: {base_cost_str}."
        }

    product_name = product_data.get(
        "product_name", product_reference
    )  # Fallback to ref if name missing

    company_data = company_margins_store.get(company_id)
    if not company_data:
        return {
            "error": f"Company with ID {company_id} not found. Cannot calculate price."
        }

    margin_percentage_val = company_data.get(
        "default_margin_percentage", 0.0
    )  # Default to 0% if not found
    # Check for product-specific margin and ensure it's a number
    if (
        "product_specific_margins" in company_data
        and product_reference in company_data["product_specific_margins"]
    ):
        specific_margin = company_data["product_specific_margins"][product_reference]
        try:
            margin_percentage_val = float(specific_margin)
        except ValueError:
            # Log an error or warning, use default if specific is invalid
            print(
                f"Warning: Invalid specific margin format for product {product_reference}, company {company_id}. Using default."
            )
            # Ensure margin_percentage_val is float if default was an int
            try:
                margin_percentage_val = float(
                    company_data.get("default_margin_percentage", 0.0)
                )
            except ValueError:  # Fallback if default_margin_percentage is also bad
                margin_percentage_val = 0.0

    # Ensure margin_percentage is float if it was loaded as int/str
    try:
        margin_percentage = float(margin_percentage_val)
    except ValueError:
        print(
            f"Warning: Invalid default margin format for company {company_id}. Using 0%."
        )
        margin_percentage = 0.0

    selling_price = base_cost * (1 + margin_percentage / 100)

    return {
        "product_reference": product_reference,
        "product_name": product_name,
        "base_cost": round(base_cost, 2),
        "margin_percentage_applied": margin_percentage,
        "calculated_selling_price": round(selling_price, 2),
        "company_id": company_id,
        "message": (
            f"For product {product_name} (Ref: {product_reference}) with company ID {company_id}, "
            f"the base cost is ${base_cost:.2f}. Applying a margin of {margin_percentage}%, "
            f"the calculated selling price is ${selling_price:.2f}."
        ),
    }


def generate_qt_prompt(product_references: List[str], company_id: str) -> dict:
    """
    Generates an initial quotation draft for specified products and a customer.
    It lists products with their stock status and margin details,
    and then prompts the user for quantity and price for each item.
    """
    company_data = company_margins_store.get(company_id)
    if not company_data:
        return {"error": f"Company with ID {company_id} not found."}
    company_name = company_data.get("company_name", f"Company ID {company_id}")

    products_info_text_parts = []
    valid_product_references_for_prompt = []

    for i, ref in enumerate(product_references):
        product_data = reference_data_store.get(ref)
        stock_info = stock_data_store.get(ref)

        if not product_data:
            products_info_text_parts.append(
                f"{i + 1}. Product {ref}: Details not found."
            )
            continue

        valid_product_references_for_prompt.append(ref)
        product_name = product_data.get("product_name", "N/A")

        margin_percentage = company_data.get("default_margin_percentage", 0)
        if (
            "product_specific_margins" in company_data
            and ref in company_data["product_specific_margins"]
        ):
            margin_percentage = company_data["product_specific_margins"][ref]

        margin_display_text = f"{margin_percentage}%"

        stock_display_text = "Stock: N/A"
        if stock_info:
            current_stock = stock_info.get("stock", 0)
            restocking_num = stock_info.get("restocking_number", 0)
            restock_date = stock_info.get("restock_date", "N/A")
            stock_display_text = f"Stock: {current_stock}"
            if current_stock == 0 and restocking_num > 0:
                stock_display_text += f" (none currently, {restocking_num} arriving around {restock_date})"
            elif restocking_num > 0:
                stock_display_text += f" (+{restocking_num} additional units arriving around {restock_date})"

        products_info_text_parts.append(
            f"{i + 1}. {product_name} (Ref: {ref}), Margin: {margin_display_text}. {stock_display_text}"
        )

    products_section = (
        "\\n".join(products_info_text_parts)
        if products_info_text_parts
        else "No product details found for the provided references."
    )

    prompt_example_ref = (
        valid_product_references_for_prompt[0]
        if valid_product_references_for_prompt
        else "XXXXX"
    )

    output_message = f"""
Company: {company_name}

Products:
{products_section}

---
To proceed with the quotation, please provide the following for each product you wish to include:
- Product Reference
- Quantity you want to sell
- Your proposed selling price per unit

For example: 'Ref {prompt_example_ref}, 10 units at $YYYY each.'
"""
    return {"quotation_prompt": output_message.strip()}


def validate_quotation_item(
    product_reference: str, company_id: str, quantity: int, proposed_price: float
) -> dict:
    """
    Validates the proposed quantity and selling price for a single product in a quotation
    against available stock and company margins.
    """
    product_data = reference_data_store.get(product_reference)
    if not product_data or "base_cost" not in product_data:
        return {
            "error": f"Product data or base cost for {product_reference} not found. Cannot validate."
        }
    base_cost = float(product_data["base_cost"])

    company_data = company_margins_store.get(company_id)
    if not company_data:
        return {
            "error": f"Company with ID {company_id} not found. Cannot validate margins."
        }

    margin_percentage = float(company_data.get("default_margin_percentage", 0))
    if (
        "product_specific_margins" in company_data
        and product_reference in company_data["product_specific_margins"]
    ):
        margin_percentage = float(
            company_data["product_specific_margins"][product_reference]
        )

    min_selling_price = base_cost * (1 + margin_percentage / 100)

    price_validation_result = {
        "status": "ok",
        "message": f"Price ${proposed_price:.2f} for {product_reference} is acceptable.",
        "recommended_min_price": round(min_selling_price, 2),
        "proposed_price": proposed_price,
        "margin_applied": f"{margin_percentage}% on base cost ${base_cost:.2f}",
    }

    if proposed_price < min_selling_price:
        price_validation_result["status"] = "too_low"
        price_validation_result["message"] = (
            f"Proposed price ${proposed_price:.2f} for {product_reference} is below the minimum. "
            f"Recommended minimum price is ${min_selling_price:.2f} "
            f"(based on {margin_percentage}% margin over cost ${base_cost:.2f})."
        )

    stock_info = stock_data_store.get(product_reference)
    current_stock = stock_info.get("stock", 0) if stock_info else 0
    restocking_num = stock_info.get("restocking_number", 0) if stock_info else 0
    restock_date = stock_info.get("restock_date", "N/A") if stock_info else "N/A"

    stock_validation_result = {
        "status": "ok",
        "message": f"Quantity {quantity} for {product_reference} is available.",
        "current_stock": current_stock,
        "requested_quantity": quantity,
    }
    if quantity <= 0:
        stock_validation_result["status"] = "invalid_quantity"
        stock_validation_result["message"] = (
            f"Requested quantity {quantity} for {product_reference} must be positive."
        )
    elif quantity > current_stock:
        stock_validation_result["status"] = "insufficient_stock"
        stock_validation_result["message"] = (
            f"Requested quantity {quantity} for {product_reference} exceeds available stock of {current_stock}. "
        )
        if restocking_num > 0:
            stock_validation_result["message"] += (
                f"{restocking_num} additional units are expected around {restock_date}."
            )
        else:
            stock_validation_result["message"] += "No restocking information available."

    return {
        "product_reference": product_reference,
        "price_validation": price_validation_result,
        "stock_validation": stock_validation_result,
    }


load_dotenv(".env.local")

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class Request(BaseModel):
    messages: List[ClientMessage]


available_tools = {
    "get_current_weather": get_current_weather,
    "get_reference_data": get_reference_data,
    "get_margins": get_margins,
    "get_customer_id_for_company": get_customer_id_for_company,
    "get_product_stock_info": get_product_stock_info,
    "generate_qt_prompt": generate_qt_prompt,
    "validate_quotation_item": validate_quotation_item,
    "get_product_price_info": get_product_price_info,
}


def do_stream(messages: List[ChatCompletionMessageParam]):
    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather at a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {
                                "type": "number",
                                "description": "The latitude of the location",
                            },
                            "longitude": {
                                "type": "number",
                                "description": "The longitude of the location",
                            },
                        },
                        "required": ["latitude", "longitude"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_reference_data",
                    "description": "Get reference data for a specific product reference ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_reference": {
                                "type": "string",
                                "description": "The reference ID of the product to look up (e.g., '00001', '00002')",
                            },
                        },
                        "required": ["product_reference"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_margins",
                    "description": "Get margin data for a specific company ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_id": {
                                "type": "string",
                                "description": "The ID of the company to look up (e.g., '1', '2')",
                            },
                        },
                        "required": ["company_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_customer_id_for_company",
                    "description": "Get a mock customer ID for a given company ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_id": {
                                "type": "string",
                                "description": "The ID of the company (e.g., '1', '2') for which to fetch the customer ID.",
                            },
                        },
                        "required": ["company_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_product_stock_info",
                    "description": "Get stock information for a specific product reference ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_reference": {
                                "type": "string",
                                "description": "The reference ID of the product to get stock information for (e.g., '00001').",
                            },
                        },
                        "required": ["product_reference"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_qt_prompt",
                    "description": "Generates an initial quotation draft. Lists products with stock status and margins, and prompts the user for quantity and price.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_references": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of product reference IDs for the quotation.",
                            },
                            "company_id": {
                                "type": "string",
                                "description": "The ID of the company for which the quotation is being generated.",
                            },
                        },
                        "required": ["product_references", "company_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_quotation_item",
                    "description": "Validates proposed quantity and selling price for a product against stock and margins.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_reference": {
                                "type": "string",
                                "description": "The reference ID of the product.",
                            },
                            "company_id": {
                                "type": "string",
                                "description": "The ID of the company (for margin lookup).",
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "The quantity of the product the user wants to sell.",
                            },
                            "proposed_price": {
                                "type": "number",
                                "description": "The selling price per unit proposed by the user.",
                            },
                        },
                        "required": [
                            "product_reference",
                            "company_id",
                            "quantity",
                            "proposed_price",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_product_price_info",
                    "description": "Calculates and returns selling price information for a product, considering the company-specific margin. If the company ID is not known from the context, the AI should ask the user for it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_reference": {
                                "type": "string",
                                "description": "The reference ID of the product (e.g., '00001', '00003').",
                            },
                            "company_id": {
                                "type": "string",
                                "description": "The ID of the company for which to calculate the price, as this determines the margin (e.g., '1', '2').",
                            },
                        },
                        "required": ["product_reference", "company_id"],
                    },
                },
            },
        ],
    )

    return stream


def stream_text(messages: List[ChatCompletionMessageParam], protocol: str = "data"):
    draft_tool_calls = []
    draft_tool_calls_index = -1

    tools_definition = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather at a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "The latitude of the location",
                        },
                        "longitude": {
                            "type": "number",
                            "description": "The longitude of the location",
                        },
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_reference_data",
                "description": "Get reference data for a specific product reference ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_reference": {
                            "type": "string",
                            "description": "The reference ID of the product to look up (e.g., '00001', '00002')",
                        },
                    },
                    "required": ["product_reference"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_margins",
                "description": "Get margin data for a specific company ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company_id": {
                            "type": "string",
                            "description": "The ID of the company to look up (e.g., '1', '2')",
                        },
                    },
                    "required": ["company_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_customer_id_for_company",
                "description": "Get a mock customer ID for a given company ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company_id": {
                            "type": "string",
                            "description": "The ID of the company (e.g., '1', '2') for which to fetch the customer ID.",
                        },
                    },
                    "required": ["company_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product_stock_info",
                "description": "Get stock information for a specific product reference ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_reference": {
                            "type": "string",
                            "description": "The reference ID of the product to get stock information for (e.g., '00001').",
                        },
                    },
                    "required": ["product_reference"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_qt_prompt",
                "description": "Generates an initial quotation draft. Lists products with stock status and margins, and prompts the user for quantity and price.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_references": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of product reference IDs for the quotation.",
                        },
                        "company_id": {
                            "type": "string",
                            "description": "The ID of the company for which the quotation is being generated.",
                        },
                    },
                    "required": ["product_references", "company_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_quotation_item",
                "description": "Validates proposed quantity and selling price for a product against stock and margins.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_reference": {
                            "type": "string",
                            "description": "The reference ID of the product.",
                        },
                        "company_id": {
                            "type": "string",
                            "description": "The ID of the company (for margin lookup).",
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "The quantity of the product the user wants to sell.",
                        },
                        "proposed_price": {
                            "type": "number",
                            "description": "The selling price per unit proposed by the user.",
                        },
                    },
                    "required": [
                        "product_reference",
                        "company_id",
                        "quantity",
                        "proposed_price",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product_price_info",
                "description": "Calculates and returns selling price information for a product, considering the company-specific margin. If the company ID is not known from the context, the AI should ask the user for it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_reference": {
                            "type": "string",
                            "description": "The reference ID of the product (e.g., '00001', '00003').",
                        },
                        "company_id": {
                            "type": "string",
                            "description": "The ID of the company for which to calculate the price, as this determines the margin (e.g., '1', '2').",
                        },
                    },
                    "required": ["product_reference", "company_id"],
                },
            },
        },
    ]

    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4.1-nano-2025-04-14",
        stream=True,
        tools=tools_definition,
    )

    for chunk in stream:
        for choice in chunk.choices:
            if choice.finish_reason == "stop":
                continue

            elif choice.finish_reason == "tool_calls":
                for tool_call in draft_tool_calls:
                    yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        args=tool_call["arguments"],
                    )

                for tool_call in draft_tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = json.loads(tool_call["arguments"])
                    tool_call_id = tool_call["id"]
                    tool_result = None
                    error_message = None

                    print(
                        f"---> [Tool Call] Executing tool: {tool_name} with args: {tool_args}"
                    )  # DEBUG

                    try:
                        # Attempt to execute the tool function
                        tool_function = available_tools[tool_name]
                        tool_result = tool_function(**tool_args)
                        print(
                            f"<--- [Tool Success] Result for {tool_name}: {tool_result}"
                        )  # DEBUG
                    except Exception as e:
                        # Capture any error during tool execution
                        print(
                            f"<--- [Tool Error] Failed tool {tool_name} with args {tool_args}: {e}"
                        )  # DEBUG Adjusted
                        error_message = str(e)

                    # Yield the result or the error
                    yield (
                        'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}'.format(
                            id=tool_call_id,
                            name=tool_name,
                            args=json.dumps(tool_args),  # Use the parsed args
                            result=json.dumps(
                                tool_result
                                if error_message is None
                                else {"error": error_message}
                            ),
                        )
                        + "\n"
                    )

            elif choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    id = tool_call.id
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments

                    if id is not None:
                        draft_tool_calls_index += 1
                        draft_tool_calls.append(
                            {"id": id, "name": name, "arguments": ""}
                        )

                    else:
                        draft_tool_calls[draft_tool_calls_index]["arguments"] += (
                            arguments
                        )

            else:
                # Only yield text content if it's not None
                if choice.delta.content is not None:
                    yield "0:{text}\n".format(text=json.dumps(choice.delta.content))

        if chunk.choices == []:
            usage = chunk.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            yield 'e:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}},"isContinued":false}}\n'.format(
                reason="tool-calls" if len(draft_tool_calls) > 0 else "stop",
                prompt=prompt_tokens,
                completion=completion_tokens,
            )


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    messages = request.messages
    openai_messages = convert_to_openai_messages(messages)

    response = StreamingResponse(stream_text(openai_messages, protocol))
    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
