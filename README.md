# Recommnder Models

## Collaborative Filtering
### The neighbourhood Approach 
[Video Link](https://www.youtube.com/watch?v=pnX6a9L6Lng)

#### Similarity metrics source
These metrics are found on those articles
* [Neighborhood Based Collaborative Filtering — Part 4](https://medium.com/fnplus/neighbourhood-based-collaborative-filtering-4b7caedd2d11#:~:text=Neighborhood%20Based%20Collaborative%20Filtering%20leverages,done%20to%20items%20as%20well!) 
*  [Collaborative-Filtering-Recommendation-Engine](https://github.com/nirmal-krishnan/Collaborative-Filtering-Recommendation-Engine)

We use those metrics to calculare the similarity between each user to our active user, then using the weighted average we can
calculate a score for every movie of the active user

Check the General algorithm on this [article](https://github.com/nirmal-krishnan/Collaborative-Filtering-Recommendation-Engine/blob/master/Final_Writeup.pdf)

##### Pearson Correlation Coefficient
![](https://miro.medium.com/max/939/1*0ev2t7hU5HqldLa9Kt-DTA.jpeg)


##### Spearman Rank Correlation Coefficient
![](https://miro.medium.com/max/658/1*rhPDyVnVJbA18V1ZVyF_dg.jpeg)


##### Mean-Squared Distance
![](https://miro.medium.com/max/875/1*olTpJjX2LKXnGdnnxXCCwg.jpeg)


##### Cosine Similarity
![](https://miro.medium.com/max/851/1*Ed681tZoQVTDsoRqAxu_-Q.jpeg)

### Matrix Facotization 
* [Part 1 Video Link](https://www.youtube.com/watch?v=h-gEB2An8bo) 
* [Part 2 Video Link]()

### Implementation


* [ ] The neighbourhood Approach
	* [x] Cosine Similarity
	* [x] Mean-Squared Distance
	* [x] Spearman Rank Correlation Coefficient
	* [x] Pearson Correlation Coefficient
	
[ ] Matrix Factorization