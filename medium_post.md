# What Are People Actually Talking About? Clustering 3,835 Forum Posts to Find a Community's Real Topic Structure

## The question, and who is asking it

**Question: when a discussion platform has a set of official boards, do the conversations actually separate along those boards, or do they collapse into a smaller number of real topic communities?**

The stakeholder is a **community operations lead** at a discussion platform, the person who owns the board taxonomy and the moderator staffing plan. They are asking because the official board list is a guess made by product, not a measured fact about how members talk.

Two decisions ride on the answer:

1. **Board consolidation or splitting.** If five official boards really only produce three distinct conversations, three of those boards should merge and the moderator hours should follow. If one board hides two separate conversations, it should split.
2. **Moderator routing.** Boards that turn out to be one blended argument culture (politics, health advice, religion) need moderators trained in contentious discussion. Boards that turn out to be clean technical support queues need moderators who can answer questions, not referee them.

This is network data in the sense that matters here: a corpus of posts from a many to many communication network, where the structure I care about is which posts sit near which other posts.

## The data

I used the **20 Newsgroups** archive, a standard public corpus of roughly 20,000 Usenet posts collected across 20 boards. Usenet is a genuine many to many discussion network and is the closest public stand in for a modern forum platform that does not require me to scrape anyone's private community.

I pulled five boards chosen to span very different conversation types:

| Board | What it is |
|---|---|
| comp.sys.mac.hardware | technical support and hardware questions |
| rec.sport.hockey | sports fandom and game recaps |
| sci.med | medical questions and clinical discussion |
| talk.politics.guns | contentious policy argument |
| soc.religion.christian | doctrinal and faith discussion |

Fields per record: the **post body**, the **board label**, and the message headers. The body is what makes this relevant to my question, because the question is about topical content. The board label is relevant in a different way: I deliberately **did not** give it to the clustering algorithm, and instead held it back as ground truth so I could check afterward whether the discovered clusters matched the official board structure. That is exactly the comparison the stakeholder is asking for.

## How I collected it

I used the archive loader in scikit learn, which downloads the corpus directly from the original distribution:

```python
from sklearn.datasets import fetch_20newsgroups

raw = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    random_state=414,
)
```

Two collection choices matter:

* **I stripped headers, signature footers, and quoted reply text.** Headers leak the board name straight into the text, which would make the clustering trivially correct and completely useless. Quoted text duplicates other posts and inflates similarity between a reply and whatever it quotes.
* **I dropped posts under 40 words.** Very short posts ("me too", "thanks, that worked") have almost no vocabulary to measure similarity on and just add noise.

That left **3,835 posts** out of 4,859.

## Measuring similarity

**Features: TF IDF weighted unigrams.** Each post becomes a vector over the vocabulary, where a term's weight rises with how often it appears in that post and falls with how many posts in the whole corpus use it. Common words that appear everywhere get crushed, distinctive words get amplified.

```python
vec = TfidfVectorizer(
    stop_words="english",
    min_df=5,          # term must appear in at least 5 posts
    max_df=0.4,        # drop terms in more than 40% of posts
    max_features=20000,
    sublinear_tf=True, # log dampen raw counts
    norm="l2",
)
```

Result: a **3,835 by 9,829** sparse matrix.

**Similarity metric: cosine similarity**, which is the angle between two post vectors. Cosine is the right choice for text because it ignores document length. A 900 word essay on goaltending and a 60 word note on goaltending should count as similar, and Euclidean distance would call them far apart purely because one vector is longer.

Because I applied L2 normalization, every vector sits on the unit sphere, and Euclidean distance becomes a monotone function of cosine distance. That means **k means on L2 normalized TF IDF is effectively spherical k means**, so I can use the fast k means implementation while still reasoning in cosine terms. I scored cluster quality with **silhouette computed on the cosine metric** to keep the evaluation consistent with the representation.

## Choosing k

I swept k from 2 to 10 and recorded mean cosine silhouette and inertia at each value.

| k | silhouette (cosine) | inertia |
|---|---|---|
| 2 | 0.0100 | 3750.8 |
| 3 | 0.0097 | 3734.1 |
| 4 | 0.0154 | 3708.5 |
| **5** | **0.0178** | **3694.6** |
| 6 | 0.0173 | 3684.3 |
| 7 | 0.0185 | 3675.2 |
| 8 | 0.0187 | 3669.3 |
| 9 | 0.0190 | 3663.6 |
| 10 | 0.0183 | 3658.5 |

*(Figure 1: silhouette curve and elbow curve, with k = 5 marked.)*

Here is the honest read. **The silhouette values are tiny, roughly 0.01 to 0.02 across the entire sweep, and the elbow curve is almost a straight line.** Neither one hands me a k. That is not a bug in my setup, it is what high dimensional sparse text does: in nearly 10,000 dimensions almost every pair of posts is close to orthogonal, so within cluster and between cluster distances end up nearly identical and silhouette is compressed toward zero for every k.

So I selected k by combining three things:

1. **The weak signal that does exist.** Silhouette rises from 0.0097 to 0.0178 between k = 3 and k = 5, then goes flat. Everything past 5 is inside the noise of that flat stretch, and the small bump at k = 9 buys 0.001 in exchange for four extra clusters.
2. **A pre defined structural hypothesis.** The stakeholder's question is explicitly "do conversations match the five official boards", so k = 5 is the hypothesis under test. Testing it requires fitting it.
3. **Interpretability at inspection time.** I fit k = 4, 5, 6, and 8 and read the top terms for each. At k = 5 every cluster had a readable identity. At k = 8 the extra clusters were slivers built on shared signature blocks rather than topics.

**I picked k = 5**, with the caveat stated plainly: this is a pre defined and interpretability driven choice that the silhouette curve permits rather than proves.

## What each cluster represents

*(Figure 2: all 3,835 posts projected into two dimensions by truncated SVD, colored by cluster.)*

### Cluster 0, "Faith and doctrine" (n = 630)

Top terms: `god, jesus, christ, christians, church, christian, people, bible, believe, sin, faith, say`

* **Example 1:** a re post of a long FAQ essay on homosexuality and Christian teaching, referred back to an archive at ftp.rutgers.edu.
* **Example 2:** a member sharing a religious tract, opening "May God richly bless those who read it."

This is doctrinal discussion. Long form, scripture referencing, in group vocabulary.

### Cluster 1, "Signature block artifact" (n = 66)

Top terms: `dsl, cadre, chastity, n3jxp, shameful, intellect, geb, skepticism, banks, surrender, pitt`

* **Example 1:** a two line answer about reflex sympathetic dystrophy.
* **Example 2:** a two line answer about eye dominance and refractive error.

**These two posts share no topic at all.** They share an author, Gordon Banks, whose email signature carries a fixed quotation about skepticism. The cluster is held together by that boilerplate, not by content. I am reporting it rather than hiding it, because it is the single most useful QA finding in the run: **this cluster is the residue of imperfect footer stripping.** For a real deployment it is also a live signal, since it shows a single prolific poster can generate enough repeated text to look like a community.

### Cluster 2, "Hardware support queue" (n = 635)

Top terms: `mac, apple, drive, monitor, thanks, use, know, does, problem, software, card, scsi`

* **Example 1:** a Macintosh general FAQ, version 2.1.3, covering FTP sites and file formats.
* **Example 2:** a comparison sheet of Mac versus IBM hardware.

This is a technical support and reference board. Note the giveaway verbs: `thanks`, `problem`, `does`. This is a question and answer queue, not a debate.

### Cluster 3, "Sports fandom" (n = 659)

Top terms: `game, team, hockey, games, play, season, players, year, nhl, espn, win, teams`

* **Example 1:** the rec.sport.hockey FAQ, listing the NHL, minor leagues, college hockey, and the Usenet hockey pool.
* **Example 2:** a game recap of the Flyers blowing a 3 goal lead to Buffalo, with Mogilny's 75th and 76th goals of the season.

Event driven fan discussion, heavy on proper nouns and numbers.

### Cluster 4, "Contested public affairs" (n = 1,845)

Top terms: `people, don, just, like, know, think, gun, time, did, good, right, ve`

* **Example 1:** a cross posted discussion of personality typing and scientific rigor, with followups directed to sci.med.
* **Example 2:** a White House press release of President Clinton's remarks and Q and A on Waco.

This is the interesting one. It is **the largest cluster by far, holding 48 percent of all posts**, and it is not one board. Its vocabulary is not topical at all, it is argumentative: `people`, `think`, `right`, `good`. This cluster is a **register**, the shared style of contested general interest debate, and it swallows most of guns, most of medicine, and a large minority of religion.

## The answer

Validated against the held out board labels, **Adjusted Rand Index = 0.477**.

| true board | c0 | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|
| comp.sys.mac.hardware | 0 | 0 | **627** | 0 | 85 |
| rec.sport.hockey | 0 | 0 | 1 | **658** | 98 |
| sci.med | 5 | 66 | 5 | 0 | **715** |
| soc.religion.christian | **623** | 0 | 2 | 0 | 243 |
| talk.politics.guns | 2 | 0 | 0 | 1 | **704** |

**Answer to the stakeholder's question: the boards only partly match the real conversations, and the mismatch is systematic rather than random.**

Read the table row by row and a clean three way split appears:

1. **Two boards are genuinely distinct communities.** Hardware is 88 percent recovered into a single cluster it shares with almost nobody. Hockey is 87 percent recovered the same way. These boards have their own vocabularies and their own reasons for posting, and they should stay separate. They are also the two boards where the conversation is a queue, so they can be staffed with subject matter moderators rather than referees.
2. **Three boards blur into one argument culture.** Guns is 91 percent inside cluster 4. Medicine is 91 percent inside cluster 4 once the signature artifact is set aside. Religion splits, with 72 percent in its own doctrinal cluster but 28 percent leaking into cluster 4. Topically these boards are unrelated. **Linguistically they are the same activity: people arguing about contested claims.** For the moderation staffing decision, that is the finding that matters. Those three boards should share a single moderation policy and a single trained pool, because the failure mode is identical even though the subject matter is not.
3. **Religion is the one board that should split.** It is the only board that produced a substantial presence in two clusters. That is a real internal division between doctrinal exposition and general argument, and it is the strongest candidate for a board split.

An ARI of 0.477 is the quantitative version of the same story: **meaningfully better than chance and nowhere near a match.** If conversations tracked boards, ARI would approach 1. If topic were unrelated to board, it would sit near 0. Landing in the middle is the actual finding, not a disappointing result.

## Validation

I did not trust the clusters on the strength of the top terms alone.

1. **A held out ground truth check.** The board label never entered the feature matrix. Computing ARI and the contingency table afterward is a real test rather than a restatement of the input, and this is what let me quantify partial agreement instead of guessing at it.
2. **Reading actual posts.** For every cluster I pulled the three posts closest to the centroid by cosine similarity and read them. This is exactly how I caught cluster 1. Its top terms (`dsl`, `cadre`, `chastity`, `n3jxp`) looked like nonsense, and reading the posts showed why: one author's email signature. **Term lists alone would have let me invent a plausible sounding story about a cluster that has no topic.**
3. **Input sanity checks.** I confirmed the retained corpus was 3,835 posts out of 4,859 after the 40 word filter, confirmed the matrix shape at 3,835 by 9,829, and verified the contingency table row sums matched the known board sizes.
4. **Fixed random seed.** Every step uses `random_state=414`, and the final fit uses `n_init=25`, so the reported numbers reproduce exactly rather than shifting between runs.

**On AI assistance:** I used an AI assistant while writing and debugging the code. I did not accept its output on trust. Every number in this post comes from the script in the repository, executed on my machine, and printed to my terminal. I re ran the pipeline end to end and confirmed the silhouette table, the ARI, and the contingency table match what is written here. When the assistant proposed metric choices I checked them against the scikit learn documentation, specifically that silhouette accepts `metric="cosine"` and that k means optimizes Euclidean distance, which is the reason I documented the L2 normalization argument instead of claiming k means "uses cosine". The AI generated cluster interpretations I discarded outright and wrote from the posts I read myself, because cluster 1 is a direct demonstration of how confident and wrong a term list based interpretation can be.

## Limitations

1. **The signature artifact means my cleaning was incomplete.** One cluster out of five, 66 posts, is a preprocessing failure and not a community. A production run needs author level signature detection before the vectorizer sees anything.
2. **Cluster 4 is under resolved.** Holding 48 percent of the corpus, it is doing too much work. The right follow up is to cluster inside it, since guns and medicine are certainly separable at a finer grain.
3. **Bag of words throws away everything except vocabulary.** Reply structure, thread membership, author, and timing all carry community signal and none of it is in my features. Two posts that argue opposite sides of the same question look nearly identical to this model, which is precisely why cluster 4 is so large.
4. **k means forces every post into exactly one cluster.** Real posts are cross posted, and one of my own example posts says so in its first line. Soft or overlapping clustering would represent that better.
5. **The corpus is biased by era and demographics.** These are Usenet posts from 1993, written by an English speaking, heavily academic and technical population with the access and inclination to be online then. Conclusions about "argument register" should not be exported to a modern platform without re running on that platform's own data.
6. **The five boards were chosen to be different.** I selected boards that span distinct domains. Real platforms have adjacent boards, and clustering would perform far worse on them. My ARI of 0.477 is therefore an optimistic upper bound.

## Code

Full pipeline, figures, and output tables: **https://github.com/Sadu3103/inst414-module3-clustering**

*Tagged: inst414smr26m03*
