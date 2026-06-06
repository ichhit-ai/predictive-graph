import pypdf
import json
import time
import re
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import google.generativeai as genai
from database import save_chunk, save_node, save_edge, save_embedding, save_nodes_batch, save_edges_batch, save_embeddings_batch, get_all_nodes, get_all_edges

# One-time Gemini API configuration
_configured_api_key = None
def configure_once(api_key):
    global _configured_api_key
    if api_key and api_key != _configured_api_key:
        genai.configure(api_key=api_key)
        _configured_api_key = api_key

# Lightweight key-term extractor for sliding context window
def extract_key_terms(text, top_n=10):
    """Extract proper-noun-like capitalized terms from text using regex."""
    # Match capitalized multi-word sequences ("C. Auguste Dupin", "Rue Morgue")
    candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    # Also match single capitalized words >= 4 chars that aren't sentence starters
    singles = re.findall(r'(?<!\. )\b[A-Z][a-z]{3,}\b', text)
    all_terms = candidates + singles
    # Count frequency, return top_n
    from collections import Counter
    counts = Counter(all_terms)
    # Filter out very common English words that happen to be capitalized
    stopwords = {'The', 'This', 'That', 'They', 'There', 'These', 'Those', 'When', 'Where',
                 'What', 'Which', 'With', 'From', 'Into', 'Upon', 'About', 'After', 'Before',
                 'Between', 'Under', 'Above', 'Such', 'Some', 'Every', 'Each', 'Both', 'Here',
                 'Have', 'Having', 'Been', 'Being', 'Were', 'Would', 'Could', 'Should', 'Will',
                 'Shall', 'Must', 'Might'}
    return [term for term, _ in counts.most_common(top_n + 10) if term not in stopwords][:top_n]

# Junk entity filter — reject obviously non-specific entities
JUNK_WORDS = {'he', 'she', 'it', 'they', 'them', 'him', 'her', 'his', 'its', 'their',
              'the', 'this', 'that', 'these', 'those', 'who', 'whom', 'which', 'what',
              'my', 'your', 'our', 'we', 'us', 'me', 'i', 'you', 'one', 'someone',
              'something', 'nothing', 'everything', 'anyone', 'everyone', 'nobody',
              'man', 'woman', 'person', 'people', 'victim', 'body', 'face', 'hand',
              'data', 'system', 'user', 'result', 'thing', 'way', 'time', 'day', 'year'}

def is_junk_entity(name):
    """Return True if the entity name is too generic or short to be useful."""
    cleaned = name.strip().lower()
    if len(cleaned) < 2:
        return True
    if cleaned in JUNK_WORDS:
        return True
    # Reject if ALL tokens are junk
    tokens = set(re.findall(r'\w+', cleaned))
    if tokens and tokens.issubset(JUNK_WORDS):
        return True
    return False

# Keyless Yahoo Search Web Scraper
def web_search(query: str, num_results: int = 3) -> str:
    """Search the web for current news, facts, or technical details.

    Args:
        query: The search terms or question to look up.
        num_results: The maximum number of search result summaries to return.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = f"https://search.yahoo.com/search?p={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        for idx, item in enumerate(soup.find_all('div', class_='algo')[:num_results]):
            h3 = item.find('h3')
            title = h3.get_text().strip() if h3 else 'Web Result'
            snippet_el = item.find('div', class_='compText')
            snippet = snippet_el.get_text().strip() if snippet_el else ''
            results.append(f"[{idx+1}] Source: {title}\nSummary: {snippet}")
        return "\n\n".join(results) if results else "No relevant search results found."
    except Exception as e:
        return f"Search error: {e}"

# Keyless Reddit Search Scraper via Yahoo Search
def reddit_search(query: str, num_results: int = 3) -> str:
    """Search Reddit for public sentiment, real conversations, or consumer feedback.

    Args:
        query: The search terms or question to look up on Reddit.
        num_results: The maximum number of Reddit search summaries to return.
    """
    return web_search(f"site:reddit.com {query}", num_results=num_results)

from collections import deque

class RollingBucketRateLimiter:
    def __init__(self, max_requests, period):
        self.max_requests = max_requests
        self.period = period
        self.requests = deque()
        self.last_request_time = 0.0
        self.min_interval = period / max_requests

    def acquire(self):
        now = time.time()
        
        # Enforce minimum spacing interval to prevent burst triggers
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            sleep_needed = self.min_interval - elapsed
            time.sleep(sleep_needed)
            now = time.time()
            
        # Clean up requests older than the sliding window period
        while self.requests and self.requests[0] <= now - self.period:
            self.requests.popleft()
        
        # If the window is full, wait until the oldest request falls out
        if len(self.requests) >= self.max_requests:
            wait_time = self.requests[0] + self.period - now
            if wait_time > 0:
                time.sleep(wait_time)
            now = time.time()
            # Clean up again after sleeping
            while self.requests and self.requests[0] <= now - self.period:
                self.requests.popleft()
        
        execution_time = time.time()
        self.requests.append(execution_time)
        self.last_request_time = execution_time

# Character-based sliding window rate limiter to respect TPM limits (e.g. 90,000 chars / 25,000 tokens per minute)
class TokenRollingBucketRateLimiter:
    def __init__(self, max_units, period=60.0):
        self.max_units = max_units
        self.period = period
        self.history = deque()
        
    def acquire(self, units):
        now = time.time()
        # Clean up history older than the sliding window
        while self.history and self.history[0][0] <= now - self.period:
            self.history.popleft()
            
        current_sum = sum(item[1] for item in self.history)
        
        # If adding these units exceeds max_units, wait until enough capacity drops out
        while current_sum + units > self.max_units:
            if not self.history:
                break
            oldest_time, oldest_units = self.history[0]
            wait_time = oldest_time + self.period - now
            if wait_time > 0:
                print(f"  [RateLimiter] TPM/Char threshold reached ({current_sum + units}/{self.max_units}). Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            now = time.time()
            # Clean up history again after waiting
            while self.history and self.history[0][0] <= now - self.period:
                self.history.popleft()
            current_sum = sum(item[1] for item in self.history)
            
        self.history.append((time.time(), units))

# Global Rate Limiters for Gemini Free Tier
# 14 RPM for LLM generation (safely under the 15 RPM limit)
llm_limiter = RollingBucketRateLimiter(max_requests=14, period=60.0)
# 99 RPM for embeddings (safely under the 100 RPM limit)
embedding_limiter = RollingBucketRateLimiter(max_requests=99, period=60.0)
# Max 90,000 characters (approx 25,000 tokens) per minute sliding window for embeddings
embedding_char_limiter = TokenRollingBucketRateLimiter(max_units=90000, period=60.0)

# Robust LLM caller with exponential backoff for rate limits (429)
def llm_call(model_name, prompt, system_instruction=None, json_mode=False, api_key=None, tools=None):
    configure_once(api_key)
    
    generation_config = {}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
        
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
        generation_config=generation_config,
        tools=tools
    )
    
    retries = 5
    delay = 2.0
    for i in range(retries):
        try:
            # Respect LLM generation rate limit
            llm_limiter.acquire()
            response = model.generate_content(prompt)
            
            # Check for safety blocks
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                raise ValueError(f"Blocked by safety: {response.prompt_feedback.block_reason}")
                
            # Check for native function calls
            try:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    part = candidate.content.parts[0]
                    if part.function_call:
                        return {
                            "function_call": {
                                "name": part.function_call.name,
                                "args": dict(part.function_call.args)
                            }
                        }
            except (AttributeError, IndexError):
                pass
                
            return response.text
        except Exception as e:
            err_msg = str(e)
            if "blocked" in err_msg.lower() or "safety" in err_msg.lower():
                raise RuntimeError(
                    "Gemini API request blocked by Safety Filters. This often happens if the "
                    "content of the document (e.g., a personal diary) contains sensitive personal reflections, "
                    "names, or topics flagged by safety policy guidelines."
                ) from e
            elif "quota" in err_msg.lower():
                raise RuntimeError(
                    "Gemini API Daily Quota Exceeded. You have used all available requests "
                    "for the day on the Free Tier (1,500 requests per day limit). "
                    "Please wait until your quota resets or supply a Paid Tier API key."
                ) from e
            elif "429" in err_msg or "ResourceExhausted" in err_msg:
                # Attempt to parse dynamic sleep time from Google gateway message
                match = re.search(r"retry in (\d+\.\d+|\d+)s", err_msg)
                sleep_time = float(match.group(1)) + 1.0 if match else delay
                time.sleep(sleep_time)
                delay *= 2
            else:
                raise e
    raise RuntimeError(
        "Gemini API rate limit exceeded (Max retries reached). You have hit the Free Tier "
        "Requests Per Minute (15 RPM) or Tokens Per Minute (TPM) limit. Please wait 60 seconds "
        "and try again, or upgrade to a Paid Tier key."
    )

# Robust embedding call with exponential backoff for rate limits
def generate_embedding_with_backoff(model_name, content, api_key=None):
    configure_once(api_key)
    retries = 5
    delay = 2.0
    char_len = len(content)
    for i in range(retries):
        try:
            # Respect character limit and request limits
            embedding_char_limiter.acquire(char_len)
            embedding_limiter.acquire()
            res = genai.embed_content(model=model_name, content=content)
            return res["embedding"]
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "ResourceExhausted" in err_msg:
                time.sleep(delay)
                delay *= 2
            else:
                raise e
    raise Exception("Max retries exceeded for embedding call.")

# Robust batch embedding call (up to 100 texts in a single call)
def generate_embeddings_batch_with_backoff(model_name, contents, api_key=None):
    if not contents:
        return []
    configure_once(api_key)
    
    chunk_size = 100
    batches = [contents[i:i + chunk_size] for i in range(0, len(contents), chunk_size)]
    all_embeddings = []
    
    for batch in batches:
        retries = 5
        delay = 2.0
        batch_res = None
        batch_chars = sum(len(text) for text in batch)
        for i in range(retries):
            try:
                # Respect character limit and request limits
                embedding_char_limiter.acquire(batch_chars)
                embedding_limiter.acquire()
                res = genai.embed_content(model=model_name, content=batch)
                batch_res = [item for item in res["embedding"]]
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "ResourceExhausted" in err_msg:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e
        if batch_res is None:
            raise Exception("Max retries exceeded for batch embedding call.")
        all_embeddings.extend(batch_res)
    return all_embeddings

# Suffixes, honorifics, and generic entities to strip for matching stems
SUFFIXES_AND_HONORIFICS = {
    'inc', 'llc', 'corp', 'co', 'corporation', 'company', 'ltd', 'limited', 'hamilton',
    'mr', 'mrs', 'ms', 'dr', 'prof', 'general', 'detective', 'inspector', 'sir', 'madam',
    'government', 'department', 'agency'
}

def entity_stem(name):
    # Lowercase, replace non-word characters
    cleaned = re.sub(r'[^\w\s]', '', name.lower())
    tokens = cleaned.split()
    # Filter out suffixes, honorifics, and single-character initials (like middle names)
    filtered = [t for t in tokens if t not in SUFFIXES_AND_HONORIFICS and len(t) > 1]
    return " ".join(filtered).strip()

# Pure Python Robust Fuzzy similarity for Entity Resolution
def token_similarity(str1, str2):
    s1 = str1.lower().replace("’", "'").replace(".", "").strip()
    s2 = str2.lower().replace("’", "'").replace(".", "").strip()
    if s1 == s2:
        return 1.0
        
    # Smarter fuzzy suffix stem matching
    stem1 = entity_stem(str1)
    stem2 = entity_stem(str2)
    if stem1 and stem2 and stem1 == stem2:
        return 1.0
        
    # Token sets
    def get_tokens(s):
        return set(re.findall(r"\w+", s))
    t1, t2 = get_tokens(s1), get_tokens(s2)
    if not t1 or not t2:
        return 0.0
        
    # Calculate Jaccard similarity of tokens
    intersection = len(t1.intersection(t2))
    union = len(t1.union(t2))
    jaccard_sim = intersection / union if union else 0.0
    
    # N-gram similarity (handles minor spelling variations like "August" / "Auguste")
    def get_ngrams(s, n=3):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
        
    ng1, ng2 = get_ngrams(s1), get_ngrams(s2)
    char_sim = 0.0
    if ng1 and ng2:
        char_sim = len(ng1.intersection(ng2)) / len(ng1.union(ng2))
        
    # If character similarity is extremely high (spelling variation), return the max
    if char_sim > 0.70:
        return max(jaccard_sim, char_sim)
        
    return jaccard_sim

def cosine_similarity(v1, v2):
    if not v1 or not v2:
        return 0.0
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_a = sum(x ** 2 for x in v1) ** 0.5
    norm_b = sum(x ** 2 for x in v2) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def retrieve_relevant_chunks(query, api_key, top_k=2):
    from database import get_all_chunks
    chunks = get_all_chunks()
    if not chunks:
        return []
        
    try:
        query_emb = generate_embedding_with_backoff("models/gemini-embedding-001", query, api_key)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return []
        
    scored_chunks = []
    for c in chunks:
        if not c.get("embedding"):
            continue
        emb = json.loads(c["embedding"])
        sim = cosine_similarity(query_emb, emb)
        scored_chunks.append((sim, c["text"]))
        
    scored_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)
    return [text for _, text in scored_chunks[:top_k]]

def retrieve_relevant_subgraph(query, api_key, top_k=15):
    """Isolate and retrieve nodes/edges most semantically relevant to the agent query."""
    from database import get_all_nodes, get_all_edges, get_all_embeddings
    nodes = get_all_nodes()
    edges = get_all_edges()
    node_embs = get_all_embeddings()
    if not nodes or not node_embs:
        return "", ""
        
    try:
        query_emb = generate_embedding_with_backoff("models/gemini-embedding-001", query, api_key)
    except Exception as e:
        print(f"Error embedding query for subgraph: {e}")
        # Fallback to top degree/centrality nodes
        nodes_str = "\n".join([f"- {n['name']} ({n['type']}, Conf: {n.get('confidence', 1.0):.2f}): {n['description']}" for n in nodes[:top_k]])
        edges_str = "\n".join([f"- {e['source']} -> {e['target']} via {e['type']} (Conf: {e.get('confidence', 1.0):.2f})" for e in edges[:top_k]])
        return nodes_str, edges_str
        
    scored_nodes = []
    for n in nodes:
        node_id = n["id"].lower()
        if node_id not in node_embs:
            continue
        sim = cosine_similarity(query_emb, node_embs[node_id])
        scored_nodes.append((sim, n))
        
    scored_nodes = sorted(scored_nodes, key=lambda x: x[0], reverse=True)
    top_nodes = [item[1] for item in scored_nodes[:top_k]]
    top_node_names = {n["name"].lower() for n in top_nodes}
    
    filtered_edges = []
    for e in edges:
        if e["source"].lower() in top_node_names and e["target"].lower() in top_node_names:
            filtered_edges.append(e)
            
    nodes_str = "\n".join([f"- {n['name']} ({n['type']}, Conf: {n.get('confidence', 1.0):.2f}): {n['description']}" for n in top_nodes])
    edges_str = "\n".join([f"- {e['source']} -> {e['target']} via {e['type']} (Conf: {e.get('confidence', 1.0):.2f})" for e in filtered_edges[:25]])
    return nodes_str, edges_str


class IngestorAgent:
    """Agent 1: Handles document parsing (pypdf) and semantic chunking."""
    def __init__(self, chunk_size=8000, chunk_overlap=1000, api_key=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.api_key = api_key

    def process_pdf(self, file_path):
        reader = pypdf.PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        # Chunking text
        chunks = []
        start = 0
        while start < len(full_text):
            end = min(start + self.chunk_size, len(full_text))
            chunks.append(full_text[start:end])
            start += self.chunk_size - self.chunk_overlap
            
        # Save chunks with embeddings to DB in batch
        try:
            embeddings = generate_embeddings_batch_with_backoff("models/gemini-embedding-001", chunks, self.api_key)
        except Exception as e:
            print(f"Error batch embedding chunks: {e}")
            embeddings = [None] * len(chunks)
            
        chunk_ids = []
        for c, emb_vector in zip(chunks, embeddings):
            chunk_ids.append(save_chunk(c, emb_vector))
            
        return chunks, chunk_ids

class ArchitectAgent:
    """Agent 2: Generates ontology, extracts triples, and resolves entities."""
    def __init__(self, api_key=None, model_name="gemini-3.1-flash-lite"):
        self.api_key = api_key
        self.model_name = model_name

    def generate_ontology(self, chunks):
        # Spaced sampling to represent the full document (start, middle, end)
        if len(chunks) >= 3:
            samples = [chunks[0], chunks[len(chunks) // 2], chunks[-1]]
        elif len(chunks) == 2:
            samples = [chunks[0], chunks[-1]]
        else:
            samples = chunks
            
        combined_samples = "\n\n---CHUNK SAMPLE---\n\n".join(samples)
        
        prompt = f"""
        Analyze the following document fragments (sampled from across the start, middle, and end of the document) and define a lightweight ontology schema.
        Output a JSON object with:
        1. "entity_types": List of entity categories (e.g. Person, Company, Technology, Concept, Event, Clue).
        2. "relation_types": List of verbs/actions describing how they link.
        
        Keep it clean, high-level, and tailored to the document content.
        
        Doc fragments:
        {combined_samples}
        """
        
        system = "You are an Ontology Architect. Generate schemas in strict JSON format."
        res = llm_call(self.model_name, prompt, system, json_mode=True, api_key=self.api_key)
        return json.loads(res)

    def extract_graph(self, chunks, ontology):
        ontology_str = json.dumps(ontology, indent=2)
        allowed_et = ", ".join(ontology.get("entity_types", []))
        allowed_rt = ", ".join(ontology.get("relation_types", []))
        
        raw_nodes = []
        raw_edges = []
        
        for idx, chunk in enumerate(chunks):
            # === SLIDING CONTEXT WINDOW ===
            # Extract key terms from up to 2 preceding chunks so cross-chunk
            # entity references aren't lost (e.g., "Dupin" mentioned in chunk 3
            # but first defined in chunk 1)
            context_terms = []
            if idx > 1:
                context_terms.extend(extract_key_terms(chunks[idx - 2], top_n=5))
            if idx > 0:
                context_terms.extend(extract_key_terms(chunks[idx - 1], top_n=8))
            # Deduplicate while preserving order
            seen = set()
            unique_context = []
            for t in context_terms:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    unique_context.append(t)
            context_str = ", ".join(unique_context) if unique_context else "(None)"
            
            prompt = f"""
            You are a thorough Knowledge Graph Builder. Extract ALL valid entities and relationships from the text below.
            
            ONTOLOGY SCHEMA (Prefer these categories, but you can add others if needed):
            Entity Types: {allowed_et}
            Relationship Types: {allowed_rt}
            
            ACTIVE CONTEXT FROM PRECEDING TEXT:
            These named entities appeared in the immediately preceding sections. If the current chunk references them (even by partial name or pronoun), resolve to these canonical forms:
            {context_str}
            
            EXTRACTION GUIDELINES:
            1. Scan the text chunk and identify EVERY named or specific entity that fits the allowed types (e.g. people, specific places, critical objects, key events, clues, organizations, concepts).
            2. Be exhaustive! Do not skip minor actors, objects, or locations. Extract as many entities as the text supports (aim for 15-30 entities per chunk if the text is dense).
            3. For each entity, write a detailed context description.
            4. Identify EVERY explicit relationship between the extracted entities. Every relationship must be directly backed by a verbatim sentence in the text as evidence.
            5. Assign a confidence score (0.0 to 1.0) to every entity and relation based on how explicitly it is stated in the text (1.0 = explicit fact, 0.5 = implied/weak connection).
            
            CRITICAL RULES:
            - DO NOT extract pronouns ("he", "she", "it", "they"). Resolve them to their proper names (e.g. write "August Dupin" instead of "he").
            - DO NOT extract generic nouns ("the victim", "the daughter", "police") when the specific named equivalent is mentioned (use "Mrs. L'Espanaye", "Camille L'Espanaye", "Isidore Muset").
            - Relationships must use precise predicates in UPPER_SNAKE_CASE (e.g., LIVES_IN, KILLED, INVESTIGATES, DISCOVERED).
            
            TEXT CHUNK:
            {chunk}
            """
            
            system = """You are a highly precise and thorough Graph Extractor. Output a strict JSON structure containing:
            {
              "entities": [{"name": "Canonical Entity Name", "type": "Entity Type", "description": "Brief context from text", "confidence": 1.0}],
              "relations": [{"source": "Source Entity Name", "target": "Target Entity Name", "type": "RELATION_TYPE", "quote": "Verbatim sentence proving connection", "confidence": 1.0}]
            }"""
            
            try:
                res = llm_call(self.model_name, prompt, system, json_mode=True, api_key=self.api_key)
                data = json.loads(res)
                
                # Add entities
                raw_nodes.extend(data.get("entities", []))
                
                # Resilient relationship mapping (accepts both relations and relationships formats)
                rel_list = data.get("relations", []) + data.get("relationships", [])
                for rel in rel_list:
                    src = rel.get("source") or rel.get("subject")
                    tgt = rel.get("target") or rel.get("object")
                    rtype = rel.get("type") or rel.get("predicate")
                    quote = rel.get("quote") or rel.get("evidence") or rel.get("context")
                    conf = rel.get("confidence")
                    try:
                        conf_val = float(conf) if conf is not None else 1.0
                    except (ValueError, TypeError):
                        conf_val = 1.0
                    if src and tgt and rtype:
                        raw_edges.append({
                            "source": str(src),
                            "target": str(tgt),
                            "type": str(rtype),
                            "quote": str(quote or ""),
                            "confidence": conf_val
                        })
            except Exception as e:
                print(f"Error extracting chunk {idx}: {e}")
                continue
                
        # === JUNK FILTERING ===
        # Remove obviously generic/pronominal entities before resolution
        filtered_nodes = [n for n in raw_nodes if not is_junk_entity(n.get("name", ""))]
        junk_count = len(raw_nodes) - len(filtered_nodes)
        if junk_count > 0:
            print(f"  [JunkFilter] Removed {junk_count} junk entities from {len(raw_nodes)} raw extractions")
        
        # Resolve entities (deduplication) using Token Jaccard Similarity
        resolved_entities = {}
        
        for n in filtered_nodes:
            name = n["name"].strip()
            if not name:
                continue
            
            # Find closest matching canonical entity
            found_match = False
            for canonical in resolved_entities:
                # If similarity threshold is > 0.65, resolve/merge
                if token_similarity(name, canonical) > 0.65 or name.lower() == canonical.lower():
                    # Merge descriptions
                    if n.get("description") and n["description"] not in resolved_entities[canonical]["description"]:
                        resolved_entities[canonical]["description"] += "; " + n["description"]
                    # Average confidence
                    n_conf = n.get("confidence")
                    try:
                        n_conf_val = float(n_conf) if n_conf is not None else 1.0
                    except (ValueError, TypeError):
                        n_conf_val = 1.0
                    resolved_entities[canonical]["confidence"] = (resolved_entities[canonical]["confidence"] + n_conf_val) / 2.0
                    found_match = True
                    break
            
            if not found_match:
                n_conf = n.get("confidence")
                try:
                    n_conf_val = float(n_conf) if n_conf is not None else 1.0
                except (ValueError, TypeError):
                    n_conf_val = 1.0
                resolved_entities[name] = {
                    "type": n["type"],
                    "description": n.get("description", ""),
                    "confidence": n_conf_val
                }
        
        # === BATCHED DATABASE WRITES ===
        # Prepare node batch
        nodes_batch = []
        texts_to_embed = []
        node_ids = []
        for name, data in resolved_entities.items():
            node_id = name.lower()
            nodes_batch.append({
                "id": node_id, 
                "name": name, 
                "type": data["type"], 
                "description": data["description"],
                "confidence": data.get("confidence", 1.0)
            })
            texts_to_embed.append(f"{name}: {data['description']}")
            node_ids.append(node_id)
            
        # Batch generate embeddings
        embeddings_batch = []
        if texts_to_embed:
            try:
                emb_vectors = generate_embeddings_batch_with_backoff("models/gemini-embedding-001", texts_to_embed, self.api_key)
                for node_id, emb_vector in zip(node_ids, emb_vectors):
                    embeddings_batch.append({"node_id": node_id, "embedding": emb_vector})
            except Exception as e:
                print(f"Error generating batch embeddings for resolved entities: {e}")
        
        # Write all nodes in one transaction
        save_nodes_batch(nodes_batch)
        
        # Prepare and write edges batch
        edges_batch = []
        for e in raw_edges:
            src, tgt = e["source"].strip(), e["target"].strip()
            if not src or not tgt:
                continue
            # Skip edges referencing junk entities
            if is_junk_entity(src) or is_junk_entity(tgt):
                continue
            
            # Find canonical names
            canonical_src = src
            canonical_tgt = tgt
            for canonical in resolved_entities:
                if token_similarity(src, canonical) > 0.65 or src.lower() == canonical.lower():
                    canonical_src = canonical
                if token_similarity(tgt, canonical) > 0.65 or tgt.lower() == canonical.lower():
                    canonical_tgt = canonical
            
            edge_id = f"{canonical_src.lower()}->{canonical_tgt.lower()}->{e['type'].lower()}"
            edges_batch.append({
                "id": edge_id, 
                "source": canonical_src, 
                "target": canonical_tgt, 
                "type": e["type"], 
                "quote": e.get("quote", ""),
                "confidence": e.get("confidence", 1.0)
            })
        
        save_edges_batch(edges_batch)
        
        # Write all embeddings in one transaction
        save_embeddings_batch(embeddings_batch)
        
        print(f"  [Graph] Saved {len(nodes_batch)} nodes, {len(edges_batch)} edges, {len(embeddings_batch)} embeddings")

class DirectorAgent:
    """Agent 3: Analyzes graph metadata, spawns 5 specialists, and runs simulations."""
    def __init__(self, api_key=None, model_name="gemini-3.1-flash-lite"):
        self.api_key = api_key
        self.model_name = model_name

    def analyze_and_spawn_swarm(self):
        nodes = get_all_nodes()
        edges = get_all_edges()
        
        # Calculate Degree Centrality to identify key entities
        centrality = {}
        for e in edges:
            centrality[e["source"]] = centrality.get(e["source"], 0) + 1
            centrality[e["target"]] = centrality.get(e["target"], 0) + 1
            
        sorted_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_entities = [item[0] for item in sorted_entities[:15]]
        
        top_nodes_data = [n for n in nodes if n["id"] in top_entities]
        nodes_str = json.dumps(top_nodes_data, indent=2)
        
        prompt = f"""
        You are the Swarm Director. Analyze this Knowledge Graph snapshot (top central nodes):
        {nodes_str}
        
        Generate exactly 5 distinct, highly specialized predictive and causal-chain forecasting personas tailored specifically to analyze and simulate outcomes for this ecosystem (e.g. if the graph is about a crime, spawn "Forensic Causal Forecaster", "Motive Profiler"; if corporate, spawn "Supply Chain Impact Forecaster", etc.). Ensure they have diverse specialties.
        
        Output a strict JSON array of objects:
        [
          {{
            "name": "Forecasting Specialist Name",
            "focus": "Specific predictive analytical focus",
            "system_prompt": "Expert guidelines instructing this agent to debate and map future risks or causal ripple effects from their domain perspective."
          }}
        ]
        """
        
        res = llm_call(self.model_name, prompt, "You are a Swarm Coordinator. Output strict JSON format.", json_mode=True, api_key=self.api_key)
        return json.loads(res)

    def run_simulation(self, scenario, specialists):
        # We simulate a 2-round debate
        debate_history = []
        debate_log = ""
        
        for round_idx in range(1, 3):
            debate_log += f"### ROUND {round_idx}\n\n"
            for agent in specialists:
                # 1. Semantic RAG retrieval of document chunks based on agent focus and scenario
                raw_excerpts = retrieve_relevant_chunks(
                    query=f"{scenario} {agent['focus']}", 
                    api_key=self.api_key, 
                    top_k=1
                )
                excerpts_str = "\n".join([f"- {text[:1200]}..." for text in raw_excerpts]) if raw_excerpts else "No direct quotes found."
                
                # 2. Dynamic, agent-specific subgraph retrieval based on agent focus and scenario
                nodes_summary, edges_summary = retrieve_relevant_subgraph(
                    query=f"{scenario} {agent['focus']}",
                    api_key=self.api_key,
                    top_k=15
                )
                if not nodes_summary:
                    nodes_summary = "No relevant entities found in graph."
                if not edges_summary:
                    edges_summary = "No relevant relationships found in graph."
                
                prompt = f"""
                Trigger Scenario: {scenario}
                
                Knowledge Graph Context (Relevant Subgraph):
                {nodes_summary}
                
                Relationships:
                {edges_summary}
                
                Raw Document Excerpt (RAG Context):
                {excerpts_str}
                
                Debate History So Far:
                {debate_log}
                
                Based on your specialized role ({agent['name']}: {agent['focus']}), analyze the graph's connections and document facts.
                
                {"ROUND 2 — ADVERSARIAL MODE: You MUST explicitly reference and CHALLENGE or COUNTER at least one specific claim made by another agent in Round 1. State which agent you disagree with and why. If you agree with all claims, identify the WEAKEST argument and stress-test it." if round_idx == 2 else ""}
                
                TOOL ACCESS:
                You have access to two tools if you need real-world data/opinions to ground your simulation:
                - web_search(query): Search the web for current news, facts, or technical details.
                - reddit_search(query): Search Reddit for public sentiment, real conversations, or consumer feedback.
                
                If you decide to use a tool, make a function call. Do not write any normal text response.
                If you do not need to search, output your constructive debate comment directly (max 4 sentences).
                """
                
                res = llm_call(
                    self.model_name, 
                    prompt, 
                    system_instruction=agent["system_prompt"], 
                    api_key=self.api_key,
                    tools=[web_search, reddit_search]
                )
                
                tool_data = ""
                tool_badge = ""
                res_str = ""
                
                if isinstance(res, dict) and "function_call" in res:
                    fc = res["function_call"]
                    name = fc["name"]
                    args = fc["args"]
                    query = args.get("query", "").strip()
                    
                    if name == "web_search":
                        search_results = web_search(query)
                        tool_data = f"\n\n[Agent Requested Web Search: '{query}']\n[Results:\n{search_results}\n]"
                        tool_badge = f"\n*(Tool Used: web_search('{query}'))*"
                    elif name == "reddit_search":
                        search_results = reddit_search(query)
                        tool_data = f"\n\n[Agent Requested Reddit Search: '{query}']\n[Results:\n{search_results}\n]"
                        tool_badge = f"\n*(Tool Used: reddit_search('{query}'))*"
                else:
                    res_str = str(res).strip()
                
                if tool_data:
                    # Query the model again with the search results
                    followup_prompt = f"""
                    Trigger Scenario: {scenario}
                    
                    You requested search results:
                    {tool_data}
                    
                    Using this external data, formulate your final predictive debate statement (max 4 sentences).
                    """
                    res = llm_call(
                        self.model_name,
                        followup_prompt,
                        system_instruction=agent["system_prompt"] + "\nFormulate your final prediction statement based on the retrieved search data.",
                        api_key=self.api_key
                    )
                    res_str = res.strip()
                
                comment = f"**{agent['name']} ({agent['focus']})**: {res_str}"
                if tool_badge:
                    comment += tool_badge
                    
                debate_log += comment + "\n\n"
                debate_history.append({"round": round_idx, "agent": agent["name"], "text": res_str + tool_badge})
                
        # Final Synthesis Pass
        synthesis_prompt = f"""
        Here is a 2-round predictive debate among 5 specialist agents about a scenario.
        
        Scenario: {scenario}
        
        Debate History:
        {debate_log}
        
        Write a final Synthesis Report containing:
        1. **Executive Prediction:** What is the most likely outcome?
        2. **Causal Chain:** Step-by-step propagation path of impact.
        3. **Confidence Score:** Percentage estimation of probability with brief reasoning.
        """
        
        synthesis = llm_call(
            "gemini-3.5-flash", # Use the stronger model for final summarization
            synthesis_prompt,
            system_instruction="You are a senior predictive intelligence forecaster. Generate markdown report.",
            api_key=self.api_key
        )
        
        # Dynamic Data Visualization Generation
        chart_info = {"needs_chart": False, "chart_description": "", "python_code": "", "success": False}
        import os
        try:
            visualizer = DataVisualizerAgent(api_key=self.api_key)
            chart_res = visualizer.generate_chart_code(scenario, debate_log)
            if chart_res.get("needs_chart"):
                scratch_dir = os.path.join(os.path.dirname(__file__), "scratch")
                os.makedirs(scratch_dir, exist_ok=True)
                output_path = os.path.join(scratch_dir, "temp_chart.png")
                
                # Delete existing temp chart if it exists
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
                
                success = execute_visualization_code(chart_res.get("python_code", ""), output_path)
                chart_info = {
                    "needs_chart": True,
                    "chart_description": chart_res.get("chart_description", ""),
                    "python_code": chart_res.get("python_code", ""),
                    "success": success
                }
        except Exception as e:
            print(f"Data Visualizer Agent execution failed: {e}")
            
        return synthesis, debate_history, chart_info


class DataVisualizerAgent:
    """Agent 4: Generates Python code using matplotlib to visualize scenario forecast data."""
    def __init__(self, api_key=None, model_name="gemini-3.1-flash-lite"):
        self.api_key = api_key
        self.model_name = model_name

    def generate_chart_code(self, scenario, debate_log):
        prompt = f"""
        You are a Data Visualization Expert. Analyze the following scenario and the multi-agent debate history.
        Determine if this report would benefit from a visual data chart (e.g. timeline probability chart, causal risk breakdown, comparative bar chart, scenario confidence projection, etc.).
        
        Scenario: {scenario}
        
        Debate History:
        {debate_log}
        
        If a visualization is NOT useful or there is no clear quantitative/timeline data, output a JSON object:
        {{
            "needs_chart": false,
            "chart_description": "Why a chart is not beneficial",
            "python_code": ""
        }}
        
        If a visualization IS useful, output a JSON object:
        {{
            "needs_chart": true,
            "chart_description": "Brief description of what the chart represents",
            "python_code": "PROPER_PYTHON_CODE_HERE"
        }}
        
        CRITICAL RULES FOR "python_code":
        1. Only use standard libraries like matplotlib.pyplot, matplotlib.ticker, and numpy.
        2. Set matplotlib backend to Agg FIRST before importing pyplot to prevent GUI crashes:
           import matplotlib
           matplotlib.use('Agg')
           import matplotlib.pyplot as plt
        3. Do NOT call plt.show(). Instead, save the plot to the predefined string variable `output_path` using:
           plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='#0d0f12')
        4. Match the theme of the Streamlit application (Dark Cyberpunk / Glassmorphic):
           - Background color of figure and axis should be '#0d0f12' or '#12161b'.
           - Label colors, tick colors, and text colors should be '#e2e8f0' or '#38bdf8'.
           - Spines (top, right) should be hidden, and bottom/left spines should be '#334155'.
           - Grid lines should be faint: plt.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
           - Use primary neon/glassmorphic plot colors: '#38bdf8' (cyan/blue), '#818cf8' (purple), '#ff4b91' (pink), '#00f5d4' (green).
        5. The variable `output_path` is already defined in the local scope, so do not redefine it.
        6. Do not include markdown code block formatting (like ```python) inside the JSON value; output the raw code as a clean string.
        """
        
        system = "You are a data visualization coder. Return a JSON object with keys 'needs_chart', 'chart_description', and 'python_code'."
        try:
            res = llm_call(self.model_name, prompt, system, json_mode=True, api_key=self.api_key)
            return json.loads(res)
        except Exception as e:
            print(f"Error generating chart code: {e}")
            return {"needs_chart": False, "chart_description": str(e), "python_code": ""}


def execute_visualization_code(code_str: str, output_path: str) -> bool:
    """Safely execute Python visualization code generated by the LLM."""
    if not code_str or not code_str.strip():
        return False
        
    # Remove markdown formatting if the model still outputs it
    clean_code = code_str.replace("```python", "").replace("```", "").strip()
    
    # Restrict execution environment for safety
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    local_scope = {
        'np': np,
        'plt': plt,
        'matplotlib': matplotlib,
        'output_path': output_path
    }
    
    try:
        # Run execution
        exec(clean_code, {}, local_scope)
        # Clear figure to release memory
        plt.close('all')
        import os
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Error executing LLM visualization code: {e}")
        try:
            plt.close('all')
        except:
            pass
        return False

