import json

from DIVO.gpt.utils import extract_code


def test_extract_code_from_plain_python_fence():
    response = "```python\ndef generate_obstacles(tblock_pose, num_obstacles):\n    return []\n```"
    code = extract_code(response)

    assert code.startswith("def generate_obstacles")
    assert "return []" in code


def test_extract_code_from_openai_compatible_json_string():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "```python\n"
                        "def generate_obstacles(tblock_pose, num_obstacles):\n"
                        "    return []\n"
                        "```"
                    )
                }
            }
        ]
    }

    code = extract_code(json.dumps(payload))

    assert code.startswith("def generate_obstacles")
    assert "return []" in code
