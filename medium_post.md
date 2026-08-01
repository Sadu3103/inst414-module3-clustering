# Five Message Boards, Three Actual Conversations

I clustered 3,835 forum posts from five message boards to see whether people talk along the lines the boards draw. Two boards hold. Three collapse into one long argument.

## The question

If a platform runs five official boards, do the conversations split five ways?

The community ops lead needs this. They own the board taxonomy and the moderator schedule, and both came from a list somebody in product invented at launch. Two decisions ride on the answer: which boards merge or split, and how to staff moderation. A support queue needs a moderator who knows the subject. An argument needs one who can referee. Different hires.

## The data

The 20 Newsgroups archive. Roughly 20,000 Usenet posts across 20 boards, public, and Usenet is a many to many discussion network, so the structure I want is in it.

I pulled five boards spread far apart: comp.sys.mac.hardware, rec.sport.hockey, sci.med, talk.politics.guns, soc.religion.christian. Each record carries a post body, a board label, and headers. The body answers the question, since the question is about what people say. The board label I held out as an answer key.

Collection is a loader call:

```python
raw = fetch_20newsgroups(subset="all", categories=CATEGORIES,
                         remove=("headers","footers","quotes"), random_state=414)
```

Headers carry the board name in plain text and quoted replies duplicate whatever they answer, so both get stripped before the model sees a word. I also dropped posts under 40 words. That took 4,859 posts to 3,835.

## Similarity

TF IDF weighted unigrams, `min_df=5`, `max_df=0.4`, sublinear term frequency, L2 normalized. Output is 3,835 by 9,829.

Distance is cosine. Length is noise in text, and a 900 word essay on goaltending belongs with a 60 word note on goaltending. K means minimizes Euclidean distance, so the L2 normalization does the work: on the unit sphere, Euclidean and cosine rank every pair identically, which makes this spherical k means. Silhouette scored on `metric="cosine"` to match.

## Picking k

| k | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| silhouette | .0100 | .0097 | .0154 | **.0178** | .0173 | .0185 | .0187 | .0190 | .0183 |

*(Figure 1: silhouette and elbow curves, k = 5 marked.)*

Everything sits between 0.01 and 0.02 and the elbow runs straight. That flatness is a property of sparse text: across 10,000 dimensions nearly every pair of documents is close to orthogonal, so silhouette compresses toward zero at every k.

Three grounds put me on 5. The signal that exists climbs to k = 5 and flattens, and the bump at k = 9 buys 0.001 for four extra clusters. The stakeholder question is whether conversations match five boards, so 5 is the hypothesis under test. And I fit 4, 5, 6, and 8 and read the top terms: at 5 every cluster had a name I could defend, by 8 the extras were slivers of shared boilerplate.

## The clusters

*(Figure 2: posts in two SVD dimensions, colored by cluster.)*

**0. Faith and doctrine, 630 posts.** god, jesus, christ, church, bible, sin, faith. A reposted FAQ essay on Christian teaching pointing to ftp.rutgers.edu. A shared tract closing with "May God richly bless those who read it."

**1. One author's email signature, 66 posts.** dsl, cadre, chastity, n3jxp, geb, skepticism, banks, pitt. Two lines on reflex sympathetic dystrophy. Two lines on eye dominance. Same author, Gordon Banks, whose footer quotes skepticism being the chastity of the intellect. The signature holds it together, and repeated boilerplate clusters as tightly as a topic does. Author level signature detection goes in the next version.

**2. Hardware support, 635 posts.** mac, apple, drive, monitor, scsi, thanks, problem. The Macintosh general FAQ version 2.1.3. A Mac versus IBM comparison sheet. Read the verbs that made the list. Thanks. Problem. Does. It is a question and answer queue.

**3. Sports, 659 posts.** game, team, hockey, nhl, espn, season. The rec.sport.hockey FAQ indexing the NHL and the Usenet hockey pool. A recap of the Flyers blowing a 3 goal lead while Mogilny scored his 75th and 76th.

**4. Contested public affairs, 1,845 posts.** people, don, just, think, gun, right, good. A cross posted thread on personality typing and scientific rigor. A White House press release of Clinton taking questions on Waco. It holds 48 percent of the corpus and its vocabulary is argumentative instead of topical. The model found a register, and it pulled in guns, medicine, and part of religion.

## The answer

Adjusted Rand Index against the held out labels: **0.477**.

| true board | c0 | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|
| comp.sys.mac.hardware | 0 | 0 | 627 | 0 | 85 |
| rec.sport.hockey | 0 | 0 | 1 | 658 | 98 |
| sci.med | 5 | 66 | 5 | 0 | 715 |
| soc.religion.christian | 623 | 0 | 2 | 0 | 243 |
| talk.politics.guns | 2 | 0 | 0 | 1 | 704 |

The boards match partway, and the mismatch follows a pattern.

Hardware lands 88 percent in one cluster, hockey 87 percent. Both carry their own vocabulary and run as queues. Keep them separate, staff them with subject moderators.

Guns sits 91 percent in cluster 4, medicine 91 percent, religion leaks 28 percent. Three unrelated subjects, one shared activity: people arguing over contested claims. One policy and one trained pool covers all three, because the failure mode is identical across them.

Religion is the board to split. It is the only one with real presence in two clusters, marking a division between members writing exposition and members arguing.

The 0.477 says it in a number. A match reads 1, no relationship reads near 0, and the middle is the finding.

## How I checked it

The board label never entered the feature matrix, so ARI and the contingency table test the clustering instead of restating it. Then I read the three posts closest to each centroid. That is how cluster 1 gave itself up: the term list looked like noise, and two posts explained it in seconds. I confirmed 3,835 posts retained of 4,859, matrix shape at 3,835 by 9,829, and contingency row sums against known board sizes. Everything runs `random_state=414` with `n_init=25`, so the numbers reproduce.

I used an AI assistant on the code and verified its output. Every number here came from the script in the repo, run on my machine, and I ran the pipeline a second time to match the silhouette table, ARI, and contingency table against what is written above. Its metric suggestions I checked against the scikit learn docs, which is why the L2 normalization argument is spelled out. Its cluster interpretations I replaced with readings of the posts, since cluster 1 shows how a term list reading sounds clean and describes nothing.

## Limits

Cluster 4 is coarse and calls for a second pass clustering inside it. Bag of words drops thread structure, reply chains, author, and timing, which is why two posts arguing opposite sides look identical to this model. K means assigns one cluster per post while forum posts get cross posted. The corpus dates to 1993 and skews academic and technical, so the register claim belongs on a modern platform only after running it there. And I picked five boards because they differ, so 0.477 reads as an optimistic ceiling.

## Code

https://github.com/Sadu3103/inst414-module3-clustering

*Tagged: inst414smr26m03*
