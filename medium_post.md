# Five Message Boards, Three Actual Conversations

I clustered 3,835 forum posts from five separate message boards to see whether people talk along the lines the boards draw. Two boards hold. Three collapse into one long argument. Here is how I got there.

## The question

If a discussion platform runs five official boards, do the conversations actually split five ways?

The person who needs this is whoever owns the board taxonomy and the moderator schedule. Call them the community ops lead. They inherited a board list somebody in product invented at launch, and nobody has checked it against how members write since.

Two decisions ride on it. Which boards merge and which split, which is a product call. And how to staff moderation, which costs money. A board that works like a support queue needs a moderator who knows the subject. A board that works like an argument needs one who can referee. Different hires, and right now the choice is a guess.

## The data

I used the 20 Newsgroups archive. About 20,000 Usenet posts across 20 boards, public, and Usenet is a many to many discussion network, so the structure I am looking for exists in it.

I pulled five boards spread as far apart as I could get them.

| Board | Conversation type |
|---|---|
| comp.sys.mac.hardware | hardware troubleshooting |
| rec.sport.hockey | sports fandom |
| sci.med | medical questions |
| talk.politics.guns | policy argument |
| soc.religion.christian | doctrine and faith |

Every record carries a post body, a board label, and headers. The body answers the question, since the question is about what people say. The board label I held out as an answer key, which is what lets me put a number on the agreement later.

Collection is a loader call against the archive.

```python
from sklearn.datasets import fetch_20newsgroups

raw = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    random_state=414,
)
```

Two decisions inside that call carry weight. Headers contain the board name in plain text and quoted reply text duplicates whatever it responds to, so both get stripped before the model sees a word. Leave either in and the clustering scores beautifully while measuring the leak. I also dropped posts under 40 words, since "thanks, that fixed it" gives me no vocabulary to compare. That took 4,859 posts down to 3,835.

## Similarity

Features are TF IDF weighted unigrams. Terms that appear across the whole corpus get flattened. Terms distinctive to a post get weight.

```python
vec = TfidfVectorizer(
    stop_words="english",
    min_df=5,
    max_df=0.4,
    max_features=20000,
    sublinear_tf=True,
    norm="l2",
)
```

Output is a 3,835 by 9,829 sparse matrix.

Distance is cosine, the angle between two post vectors. Length is noise in text. A 900 word essay on goaltending and a 60 word note on goaltending belong to the same conversation, and Euclidean distance separates them on size alone.

One mechanical point worth stating. K means minimizes Euclidean distance and takes no cosine metric. The L2 normalization above handles it. Once every vector sits on the unit sphere, Euclidean and cosine distance rank every pair the same way, so k means on normalized TF IDF runs as spherical k means. I scored quality with silhouette on `metric="cosine"` to keep the evaluation consistent with the features.

## Picking k

I swept k from 2 to 10.

| k | silhouette (cosine) | inertia |
|---|---|---|
| 2 | 0.0100 | 3750.8 |
| 3 | 0.0097 | 3734.1 |
| 4 | 0.0154 | 3708.5 |
| 5 | 0.0178 | 3694.6 |
| 6 | 0.0173 | 3684.3 |
| 7 | 0.0185 | 3675.2 |
| 8 | 0.0187 | 3669.3 |
| 9 | 0.0190 | 3663.6 |
| 10 | 0.0183 | 3658.5 |

*(Figure 1: silhouette and elbow curves, k = 5 marked.)*

Every silhouette value sits between 0.01 and 0.02. The elbow runs almost straight. That flatness is a property of sparse text in high dimensions: across roughly 10,000 dimensions nearly every pair of documents comes out close to orthogonal, so the distance to your own cluster and the distance to a neighboring one land in the same range and silhouette compresses toward zero at every k.

So I selected k on three grounds.

The weak signal that does exist climbs from k = 3 to k = 5 and flattens after. Everything past 5 sits inside that flat stretch, and the bump at k = 9 buys 0.001 for four extra clusters.

The stakeholder question is whether conversations match five boards, so 5 is the hypothesis under test and fitting it is the test.

And interpretability decided it. I fit 4, 5, 6, and 8, then read the top terms for each. At k = 5 every cluster carried a name I could defend. By k = 8 the extra clusters were slivers built on shared boilerplate.

k = 5, chosen on structure and interpretability, with the silhouette curve permitting the choice.

## What the clusters are

*(Figure 2: posts projected to two dimensions with truncated SVD, colored by cluster.)*

**Cluster 0, faith and doctrine, 630 posts.** Top terms: god, jesus, christ, christians, church, christian, people, bible, believe, sin, faith, say. One member reposts a long FAQ essay on homosexuality and Christian teaching and points readers to an archive at ftp.rutgers.edu. Another shares a religious tract and closes with "May God richly bless those who read it." Long form, scripture heavy, written for readers already inside the vocabulary.

**Cluster 1, one author's email signature, 66 posts.** Top terms: dsl, cadre, chastity, n3jxp, shameful, intellect, geb, skepticism, banks, surrender, pitt. One post is two lines on reflex sympathetic dystrophy. Another is two lines on eye dominance and refractive error. What they share is an author, Gordon Banks, whose signature block carries a fixed quote about skepticism being the chastity of the intellect. The footer holds the cluster together.

That finding is the most useful thing in the run. Sixty six posts of repeated boilerplate cluster as tightly as a topic does, which means one prolific poster with a distinctive signature can manufacture a community that was never there. Author level signature detection goes into the next version of the pipeline, ahead of the vectorizer.

**Cluster 2, hardware support, 635 posts.** Top terms: mac, apple, drive, monitor, thanks, use, know, does, problem, software, card, scsi. The two posts nearest the centroid are the Macintosh general FAQ version 2.1.3 and a Mac versus IBM comparison sheet. Read the verbs that made the top terms. Thanks. Problem. Does. That is a question and answer queue.

**Cluster 3, sports, 659 posts.** Top terms: game, team, hockey, games, play, season, players, year, nhl, espn, win, teams. Nearest posts are the rec.sport.hockey FAQ, which indexes the NHL, minor leagues, college hockey and the Usenet hockey pool, and a recap of the Flyers blowing a 3 goal lead to Buffalo while Mogilny scored his 75th and 76th of the season. Proper nouns and numbers throughout.

**Cluster 4, contested public affairs, 1,845 posts.** Top terms: people, don, just, like, know, think, gun, time, did, good, right, ve. Nearest posts are a cross posted thread on personality typing and scientific rigor, and a White House press release of Clinton taking questions on Waco.

It holds 48 percent of the corpus, and its vocabulary is argumentative. People. Think. Right. Good. The model found a register here, the shared style of contested general interest debate, and that register pulled in most of guns, most of medicine, and a slice of religion.

## The answer

Against the held out labels, Adjusted Rand Index came out at 0.477.

| true board | c0 | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|
| comp.sys.mac.hardware | 0 | 0 | 627 | 0 | 85 |
| rec.sport.hockey | 0 | 0 | 1 | 658 | 98 |
| sci.med | 5 | 66 | 5 | 0 | 715 |
| soc.religion.christian | 623 | 0 | 2 | 0 | 243 |
| talk.politics.guns | 2 | 0 | 0 | 1 | 704 |

The boards match the conversations partway, and the mismatch follows a pattern.

Two boards stand on their own. Hardware lands 88 percent in a single cluster it shares with almost nobody. Hockey lands 87 percent the same way. Both carry their own vocabulary and their own reason for posting. Keep them separate and staff them with moderators who know the subject, because both run as queues.

Three boards fuse into one argument culture. Guns sits 91 percent inside cluster 4. Medicine sits 91 percent inside cluster 4 once the signature cluster is set aside. Religion splits, 72 percent doctrinal and 28 percent into cluster 4. Three unrelated subjects, one shared activity: people arguing over contested claims. For the staffing decision that is the finding. One policy and one trained pool covers all three, because the moderation failure mode is identical across them.

Religion is the board to split. It is the only one with real presence in two clusters, which marks an internal division between members writing exposition and members arguing.

The 0.477 says the same thing in one number. A perfect match reads 1, an unrelated pair reads near 0, and the middle is the finding.

## How I checked it

Holding out the board label did the heavy lifting. It never entered the feature matrix, so computing ARI and the contingency table afterward tests the clustering instead of restating it, and it is what turned "partway" into a number.

Then I read posts. For every cluster I pulled the three closest to the centroid by cosine similarity and read them straight through. That is how cluster 1 gave itself up. Its term list looked like noise, and two posts explained it in seconds. Term lists on their own would have supported a confident paragraph about a cluster with no topic in it.

The rest is bookkeeping. 3,835 posts retained out of 4,859 after the length filter. Matrix shape confirmed at 3,835 by 9,829. Contingency row sums checked against the known board sizes. Everything runs on `random_state=414` with `n_init=25` on the final fit, so the numbers here reproduce exactly.

On AI assistance: I used an assistant while writing and debugging the code, and I verified its output. Every number in this post came out of the script in the repo, run on my machine. I ran the pipeline start to finish a second time and matched the silhouette table, the ARI, and the contingency table against what is written above. When it proposed metric choices I checked them against the scikit learn documentation, which is where the k means and cosine mechanics above came from, and it is why the L2 normalization argument is spelled out instead of a loose claim that k means uses cosine. Its cluster interpretations I replaced with readings of the posts themselves, since cluster 1 shows exactly how a term list reading can sound clean and describe nothing.

## Limits of this analysis

Cluster 4 is coarse. Half the corpus in one bucket calls for a second pass, clustering inside it, since guns and medicine separate at a finer grain.

Bag of words keeps vocabulary and drops everything else. Thread structure, reply chains, author, and timing all carry community signal, and none of it reaches the features. Two posts arguing opposite sides of one question look nearly identical to this model, which explains much of cluster 4's size.

K means assigns every post to exactly one cluster, and forum posts get cross posted. One of my own example posts announces its cross post in the first line. Soft clustering represents that better.

The corpus dates to 1993 and comes from an English speaking, heavily academic and technical population, since those were the people online. A claim about argument register belongs on a modern platform only after running the pipeline on that platform's own data.

And I chose five boards specifically because they differ from each other. Real platforms run adjacent boards that overlap in subject. Clustering scores lower there, so read 0.477 as an optimistic ceiling.

## Code

Full pipeline, figures, and output tables: https://github.com/Sadu3103/inst414-module3-clustering

*Tagged: inst414smr26m03*
