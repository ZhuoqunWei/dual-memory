# Publishing

This repo is ready to publish once GitHub CLI authentication is refreshed.

## Prerequisites

```bash
gh auth login -h github.com
gh auth status
```

`gh auth status` should show an authenticated `github.com` account before publishing.

## Create A New Public Repository

From the repository root:

```bash
gh repo create dual-memory \
  --public \
  --description "Dual-agent knowledge graph memory system for AI assistants" \
  --source . \
  --remote origin \
  --push
```

Then add useful repository topics in GitHub:

```text
ai-memory
knowledge-graph
neo4j
openclaw
typescript
python
llm
retrieval
```

## Push To An Existing Repository

If the GitHub repository already exists, add its remote and push:

```bash
git remote add origin git@github.com:ZhuoqunWei/dual-memory.git
git push -u origin master
```

Use the HTTPS remote instead if your GitHub account is configured for HTTPS:

```bash
git remote add origin https://github.com/ZhuoqunWei/dual-memory.git
git push -u origin master
```

## After Publishing

1. Confirm the README renders cleanly on GitHub.
2. Confirm the `Check` workflow passes.
3. Load `examples/demo-seed.cypher` locally and capture a graph screenshot if you want a visual for the repo page or portfolio.
4. Pin the repository on your GitHub profile.
