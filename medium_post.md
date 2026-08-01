# Five Message Boards, Three Actual Conversations

I went into this expecting a boring result. Cluster a bunch of forum posts from five very different message boards, watch five clean clusters fall out, write it up, done. That is not what happened, and the part that broke is the part I ended up caring about.

## The question

If a discussion platform runs five official boards, do the conversations actually split five ways?

The person who needs to know this is whoever owns the board taxonomy and the moderator schedule. Call them the community ops lead. They inherited a board list that somebody in product invented at launch, and they have never checked whether members actually talk along those lines.

Two things depend on the answer. First, whether any boards should merge or split, which is a straight product decision. Second, and more expensive, how to staff moderation. A board that turns out to be a support queue needs someone who knows the subject. A board that turns out to be an argument needs someone who can referee one. Those are different hires, and right now the ops lead is guessing.

## What I used

The 20 Newsgroups archive. It is about 20,000 Usenet posts across 20 boards, and it is public, which mattered to me because the alternative was scraping somebody's live community. Usenet is a real many to many discussion network, so the structure I am looking for is genuinely there.

I took five boards, picked to be as far apart as I could get them:

| Board | Conversation type |
|---|---|
| comp.sys.mac.hardware | hardware troubleshooting |
| rec.sport.hockey | sports fandom |
| sci.med | medical questions |
| talk.politics.guns | policy argument |
| soc.religion.christian | doctrine and faith |

Each record has a post body, a board label, and headers. The body is the whole point, since the question is about what people say. The board label I deliberately kept out of the model and saved as an answer key, which is the only reason I can say anything quantitative later about how well the clusters lined up.

Collection was a loader call:

```python
from sklearn.datasets import fetch_20newsgroups

raw = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    random_state=414,
)
```

Two decisions in there matter more than they look. Stripping headers and quoted text is mandatory, because the headers literally contain the board name and the quoted text duplicates whatever a reply is responding to. Leave either one in and the clustering looks brilliant while measuring nothing. I also dropped anything under 40 words, since "thanks, that fixed it" carries no vocabulary to compare against. That took me from 4,859 posts down to 3,835.

## Similarity

Features are TF IDF weighted unigrams. Words that show up everywhere get flattened, words that are distinctive to a post get weight.

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

That gives me 3,835 posts by 9,829 terms.

For distance I used cosine, the angle between two post vectors. Text needs cosine because length is noise here. Somebody's 900 word essay about goaltending and somebody's 60 word note about goaltending are the same conversation, and Euclidean distance would put them far apart just because one vector is bigger.

There is a wrinkle worth being upfront about. K means minimizes Euclidean distance, so it does not take a cosine metric. What saves it is the L2 normalization above: once every vector sits on the unit sphere, Euclidean distance and cosine distance rank pairs identically, so k means on normalized TF IDF is spherical k means in practice. I scored quality with silhouette using `metric="cosine"` to stay consistent with how I built the features.

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

Those silhouette numbers are terrible, and the elbow is basically a ramp. Neither curve told me anything.

I spent a while assuming I had broken something. I had not. In roughly 10,000 dimensions almost every pair of documents is close to orthogonal, so the distance to your own cluster and the distance to a different cluster come out nearly the same, and silhouette gets squashed toward zero no matter what k you feed it. This is a known property of sparse text, and once I stopped waiting for the curve to hand me an answer I picked k a different way.

Three things pushed me to 5. The weak signal that does exist rises from k = 3 to k = 5 and then flattens, so everything past 5 is inside the noise, and the tiny bump at k = 9 costs four extra clusters to buy 0.001 of silhouette. The stakeholder question is specifically "do conversations match the five boards", so 5 is the hypothesis I am here to test. And when I fit 4, 5, 6, and 8 and read the top terms for each, k = 5 was where every cluster had an identity I could name. By k = 8 the extras were slivers built on shared boilerplate.

So k = 5, chosen on structure and interpretability, with the silhouette curve permitting it rather than proving it. I would rather say that plainly than dress up a 0.018 as evidence.

## What the clusters are

*(Figure 2: posts projected to two dimensions with truncated SVD, colored by cluster.)*

**Cluster 0, faith and doctrine, 630 posts.** Top terms: god, jesus, christ, christians, church, christian, people, bible, believe, sin, faith, say. One member reposts a long FAQ essay on homosexuality and Christian teaching, pointing readers to an archive at ftp.rutgers.edu. Another shares a religious tract and closes with "May God richly bless those who read it." Long form, scripture heavy, written for people already inside the vocabulary.

**Cluster 1, 66 posts, and this one is a mess.** Top terms: dsl, cadre, chastity, n3jxp, shameful, intellect, geb, skepticism, banks, surrender, pitt. One post is two lines about reflex sympathetic dystrophy. Another is two lines about eye dominance and refractive error. They have nothing in common topically. What they share is an author, Gordon Banks, whose email signature carries a fixed quote about skepticism being the chastity of the intellect. The cluster is held together by his footer.

I am leaving it in the writeup because finding it was the most useful thing that happened all run. My footer stripping was incomplete, and 66 posts of residue was enough to look like a community. It also says something real about forum data: one prolific poster with a distinctive signature can manufacture a fake topic.

**Cluster 2, hardware support, 635 posts.** Top terms: mac, apple, drive, monitor, thanks, use, know, does, problem, software, card, scsi. The two posts nearest the centroid are the Macintosh general FAQ version 2.1.3 and a Mac versus IBM comparison sheet. Look at the verbs that made the list. Thanks. Problem. Does. This is a question and answer queue.

**Cluster 3, sports, 659 posts.** Top terms: game, team, hockey, games, play, season, players, year, nhl, espn, win, teams. Nearest posts are the rec.sport.hockey FAQ, which indexes the NHL, minor leagues, college hockey and the Usenet hockey pool, and a recap of the Flyers blowing a 3 goal lead to Buffalo while Mogilny scored his 75th and 76th of the season. Proper nouns and numbers all the way down.

**Cluster 4, 1,845 posts, and this is where it gets interesting.** Top terms: people, don, just, like, know, think, gun, time, did, good, right, ve. Nearest posts are a cross posted thread about personality typing and scientific rigor, and a White House press release of Clinton taking questions on Waco.

It holds 48 percent of the corpus. And its vocabulary is barely topical, it is argumentative. People. Think. Right. Good. What the model found here is a register rather than a subject, the shared style of contested general interest debate, and it pulled in most of guns, most of medicine, and a chunk of religion.

## The answer

Against the answer key I held back, Adjusted Rand Index came out at 0.477.

| true board | c0 | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|
| comp.sys.mac.hardware | 0 | 0 | 627 | 0 | 85 |
| rec.sport.hockey | 0 | 0 | 1 | 658 | 98 |
| sci.med | 5 | 66 | 5 | 0 | 715 |
| soc.religion.christian | 623 | 0 | 2 | 0 | 243 |
| talk.politics.guns | 2 | 0 | 0 | 1 | 704 |

The boards match the real conversations partially, and the mismatch follows a pattern.

Two boards hold up on their own. Hardware lands 88 percent in one cluster it shares with almost nobody, hockey lands 87 percent the same way. They have their own vocabularies and their own reasons for posting. Keep them separate, staff them with people who know the subject, because both are queues.

Three boards collapse into one argument culture. Guns is 91 percent inside cluster 4. Medicine is 91 percent inside cluster 4 once you set the signature artifact aside. Religion splits, 72 percent doctrinal and 28 percent leaking into cluster 4. Topically these three have nothing to do with each other. Linguistically they are the same activity, which is people arguing about contested claims, and for a staffing decision that is the finding that matters. One policy, one trained pool. The subject matter differs but the failure mode is identical.

Religion is the one board I would actually split. It is the only board with a real presence in two clusters, which points to an internal division between people writing exposition and people arguing.

The 0.477 is the same story in one number. Perfect agreement would be 1, no relationship would be near 0, and landing in the middle is the result rather than a disappointment.

## How I checked it

Holding out the board label was the main thing. It never touched the feature matrix, so computing ARI and the contingency table afterward is a test instead of a restatement, and it is what let me quantify partial agreement.

Beyond that, I read posts. For every cluster I pulled the three closest to the centroid by cosine similarity and actually read them, which is exactly how cluster 1 fell apart. Its term list looked like noise, and reading two posts explained it in about ten seconds. Term lists alone would have let me write a confident paragraph about a cluster that has no topic in it.

The rest is bookkeeping. I confirmed 3,835 posts retained out of 4,859 after the length filter, confirmed the matrix came out 3,835 by 9,829, and checked that the contingency row sums matched the known board sizes. Everything runs on `random_state=414` with `n_init=25` on the final fit, so the numbers here reproduce instead of drifting between runs.

On AI assistance: I used an assistant while writing and debugging the code, and I checked its work rather than trusting it. Every number in this post came out of the script in the repo, run on my machine. I ran the pipeline start to finish again and matched the silhouette table, the ARI, and the contingency table against what is written above. Where the assistant suggested metric choices I checked them against the scikit learn docs, which is how the k means and cosine wrinkle above got documented properly instead of me writing "k means with cosine distance" and being wrong. The cluster interpretations it offered I threw out and rewrote from the posts I read, since cluster 1 is a live demonstration of how wrong a term list reading can be while sounding fine.

## Where this falls short

The signature artifact means my cleaning was not finished. One cluster in five is preprocessing residue. A real run needs author level signature detection before the vectorizer sees anything.

Cluster 4 is under resolved. Half the corpus in one bucket is too much, and the obvious follow up is to cluster inside it, since guns and medicine have to be separable at a finer grain.

Bag of words throws out almost everything. Thread structure, who replied to whom, author, timing, all of it carries community signal and none of it is in my features. Two posts arguing opposite sides of the same question look nearly identical to this model, which is a large part of why cluster 4 got so big.

K means also forces every post into exactly one cluster, and real posts get cross posted. One of my own example posts announces that it is cross posted in its first line. Soft clustering would handle that better.

The corpus is from 1993, written by an English speaking, heavily academic and technical population, since those were the people online. I would not export a claim about "argument register" to a modern platform without running it on that platform's own data.

Last one, and it cuts against my own result: I chose five boards specifically because they were different from each other. Real platforms have adjacent boards that overlap. Clustering would do considerably worse there, so 0.477 should be read as an optimistic ceiling.

## Code

Full pipeline, figures, and output tables: https://github.com/Sadu3103/inst414-module3-clustering

*Tagged: inst414smr26m03*
