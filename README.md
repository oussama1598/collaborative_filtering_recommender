# Recommnder Models

## Collaborative Filtering
### The neighbourhood Approach 
[Video Link](https://www.youtube.com/watch?v=pnX6a9L6Lng)

#### General Algorithm
- Compute the similarity between our active user against every other user on our database using the one of the similarity methods
- Sort the similarities
- Use the top N neighbours as weight to compute the the score for a certain movie of our user

Check the General algorithm on this [article](https://github.com/nirmal-krishnan/Collaborative-Filtering-Recommendation-Engine/blob/master/Final_Writeup.pdf)

#### Similarity methods
- Pearson Correlation Coefficient
- Spearman Rank Correlation Coefficient
- Mean-Squared Distance
- Cosine Similarity

More details about them on [here](https://github.com/nirmal-krishnan/Collaborative-Filtering-Recommendation-Engine/blob/master/Final_Writeup.pdf)

### Matrix Facotization 
* [Part 1 Video Link](https://www.youtube.com/watch?v=h-gEB2An8bo) 
* [Part 2 Video Link]()

### Implementation
* [x] The neighbourhood Approach
	* [x] Cosine Similarity
	* [x] Mean-Squared Distance
	* [x] Spearman Rank Correlation Coefficient
	* [x] Pearson Correlation Coefficient
	
[ ] Matrix Factorization