from typing import List

import openai

def query_llm(
    client,
    system_message: str,
    user_message: str,
    user_images: List,
):
    # system input
    system_input = {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": system_message,
            }
        ]
    }

    # user input
    user_text_content = [
        {
            "type": "input_text",
            "text": user_message,
        },
    ]
    user_image_content = [
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{img}",
        } for img in user_images
    ]
    user_input = {
        "role": "user",
        "content": user_text_content + user_image_content
    }

    # send query
    response = client.responses.create(
        model="gpt-4o",
        input=[system_input, user_input],
        reasoning={},
        tools=[],
        temperature=0,
        max_output_tokens=5000,
        top_p=1,
        store=True
    )

    return response