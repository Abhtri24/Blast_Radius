path = "docker-compose.yml"
with open(path, encoding="utf-8") as f:
    content = f.read()

old = '''  app_service:
    build: ./services/app_service
    environment:
      - RETRIEVAL_URL=http://retrieval_service:8000
      - LLM_URL=http://llm_service:8000
    depends_on:
      - retrieval_service
      - llm_service
    ports:
      - "8000:8000"'''

new = '''  app_service:
    build: ./services/app_service
    environment:
      - RETRIEVAL_URL=http://retrieval_service:8000
      - LLM_URL=http://llm_service:8000
      - SOURCEGRAPH_URL=http://host.docker.internal:7080
      - SOURCEGRAPH_TOKEN=${SOURCEGRAPH_TOKEN:-}
      - SOURCEGRAPH_REPO=${SOURCEGRAPH_REPO:-github.com/Abhtri24/Backend-Project}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      - retrieval_service
      - llm_service
    ports:
      - "8000:8000"'''

assert content.count(old) == 1, f"expected 1, found {content.count(old)}"
content = content.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("patched OK")