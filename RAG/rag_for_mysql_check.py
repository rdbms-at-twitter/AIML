import warnings
warnings.filterwarnings("ignore", module="boto3")

import boto3
import json
import sys
from botocore.config import Config

config = Config(read_timeout=60)
bedrock_agent = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1', config=config)

query = sys.argv[1] if len(sys.argv) > 1 else input("\n💬 質問を入力: ")

if not query.strip():
    print("❌ 質問が入力されていません。")
    sys.exit(0)

print(f"\n{'='*60}")
print(f"📝 質問: {query}")
print(f"{'='*60}")

# Step 1: Retrieve
print("\n🔍 Knowledge Base を検索中...")
response = bedrock_agent.retrieve(
    knowledgeBaseId='UJWVXAAAAA',
    retrievalQuery={'text': query}
)

# Step 2: スコア表示と判定
print(f"\n📊 検索結果 (上位{len(response['retrievalResults'])}件):")
print(f"{'─'*60}")
for i, r in enumerate(response['retrievalResults'], 1):
    score = r['score']
    bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
    mark = "✅" if score > 0.5 else "  "
    print(f"  {mark} [{i}] {bar} {score:.3f} | {r['content']['text'][:60].strip()}")
print(f"{'─'*60}")
print(f"  閾値: 0.5 | ✅ = RAG採用チャンク")

relevant = [r for r in response['retrievalResults'] if r['score'] > 0.5]

if relevant:
    source = "RAG (Knowledge Base)"
    context = "\n\n".join([r['content']['text'][:500] for r in relevant[:3]])
    prompt = ("以下のコンテキストに基づいて回答してください。"
              "コンテキストに情報がない場合は、あなたの知識で補完して回答し、"
              "その旨を明記してください。\n\n"
              "Context:\n" + context + "\n\nQuestion: " + query)
else:
    source = "LLM直接 (モデル知識)"
    prompt = query

# Step 3: LLM呼び出し
print(f"\n🤖 モデルに問い合わせ中... [{source}]")
body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": prompt}]
})

model_response = bedrock.invoke_model(
    modelId='us.anthropic.claude-sonnet-4-6',
    body=body
)

result = json.loads(model_response['body'].read())
answer = result['content'][0]['text']

print(f"\n{'='*60}")
print(f"💡 回答 [{source}]")
print(f"{'='*60}\n")
print(answer)
print(f"\n{'='*60}")

