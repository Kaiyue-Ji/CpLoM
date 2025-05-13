import openai
from openai import OpenAI
import httpx

client = OpenAI(
    base_url="https://api.gpts.vin/v1",
    api_key="sk-okfpmnWIGjFSft3k3a3713Ce0d2641538a5cD902F11b7a11",
    http_client=httpx.Client(
        base_url="https://api.gpts.vin/v1",
        follow_redirects=True,
    ),
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion)


# def query(message,model):
#     '''
#     Query ChatGPT API
#     :param message:
#     :return:
#     '''
#     client = OpenAI(
#         base_url="https://api.gpts.vin",
#         api_key="sk-okfpmnWIGjFSft3k3a3713Ce0d2641538a5cD902F11b7a11",
#         http_client=httpx.Client(
#             base_url="https://api.gpts.vin",
#             follow_redirects=True,
#         ),
#     )
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "Hello!"}
#         ],
#     )
# # 假设 completion 是你的 ChatCompletion 对象
#     if response.choices:
#         first_choice = response.choices[0]  # 访问第一个 Choice 对象
#         message_content = first_choice.message.content  # 访问第一个 Choice 对象的 message.content
#         print(message_content)  # 打印生成的文本
#     else:
#         print("没有可用的选择")
# # result = response["choices"][0]["message"]["content"].strip()
