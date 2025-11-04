# Project-4 Clustering

The goal of this project is to analyze customer purchasing behavior in an online retail store and segment customers into distinct groups using K-Means clustering. Retail businesses often have large volumes of transactional data but lack insight into how different types of customers behave. By applying clustering, we can uncover patterns that are not directly visible through manual analysis.

The main problem this project addresses is how to identify distinct customer groups based on their purchasing habits. Understanding these groups helps businesses tailor marketing strategies, improve customer retention, and increase overall sales efficiency.

This project aims to answer the following key questions:

<ul>
    <li>Can customers be grouped based on how recently, how often, and how much they purchase?</li>
    <li>What are the characteristics of each customer segment (e.g., high-value vs. low-value customers)?</li>
    <li>How can these insights be used to guide business decisions, such as targeted marketing or loyalty programs?</li>
</ul>

By clustering customers using Recency, Frequency, and Monetary (RFM) metrics, the project provides a data-driven foundation for strategic customer relationship management.

## What is Clustering?

Clustering is an unsupervised machine learning technique used to group similar data points together based on their features. The goal is to ensure that data points within the same group (called a cluster) are more similar to each other than to those in other clusters. Unlike classification, clustering does not use predefined labels.

### K-Means Clustering

K-Means is one of the most widely used clustering algorithms. It works as follows:

<ol>
    <li>
    <strong>Choose k (number of clusters).</strong>
    <br> You decide how many clusters you want to create.
    </li>
    <li>
    <strong>Initialize centroids.</strong>
    <br> Randomly select k points (called centroids) as the initial centers of the clusters.
    </li>
    <li>
    <strong>Assign points to nearest centroid.</strong>
    <br> Each data point is assigned to the cluster whose centroid is closest (usually by Euclidean distance).
    </li>
    <li>
    <strong>Update centroids.</strong>
    <br> Compute new centroids by taking the mean of all points in each cluster.
    </li>
    <li>
    <strong>Repeat until convergence.</strong>
    <br> The assignment and update steps repeat until the centroids stop changing significantly
    </li>
</ol>

## Data Introduction

The dataset used in this project is the Online Retail Dataset from the UCI Machine Learning Repository.(<a href="https://archive.ics.uci.edu/ml/datasets/online+retail">Link</a>) This dataset contains real-world transactional data from a UK-based online retail company that sells giftware. The data covers transactions occurring between December 2010 and December 2011. Each row in the dataset represents a single product purchased in an order.

The dataset is commonly used for customer segmentation, market basket analysis, and sales trend analysis.

### Data Attributes

<ol>
    <li>
        <strong>InvoiceNo:</strong>
        a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation
    </li>
    <li>
        <strong>StockCode:</strong>
        a 5-digit integral number uniquely assigned to each distinct product
    </li>
    <li>
        <strong>Description:</strong>
        product name
    </li>
    <li>
        <strong>Quantity:</strong>
        the quantities of each product (item) per transaction
    </li>
    <li>
        <strong>InvoiceDate</strong>
        the day and time when each transaction was generated
    </li>
    <li>
        <strong>UnitPrice:</strong>
        product price per unit
    </li>
    <li>
        <strong>CustomerID:</strong>
        a 5-digit integral number uniquely assigned to each customer
    </li>
    <li>
        <strong>Country:</strong>
        the name of the country where each customer resides
    </li>
</ol>

## Data Understanding

<img src="heatmap.png" alt="heatmap for recency, frequency and monetary">

## Data Pre-Processing

### Handling Missing Value

Many transactions have no CustomerID (anonymous buyers). These rows cannot be linked to a customer, so they are removed to ensure accurate RFM calculations.

### Remove Invalid Transactions

Some records represent returns or cancellations (negative quantities). These do not represent actual purchases and would distort spending and frequency calculations.

### Create Total Purchase Value

The dataset doesn’t include a total order amount per transaction. TotalPrice gives the total monetary value per line item, which is required for computing the Monetary feature.

### Convert Dates to Datetime Format

Ensures consistent date handling and allows calculation of Recency (days since last purchase).

### Aggregate to Customer Level (RFM Calculation)

Clustering is done per customer, not per transaction. Aggregating converts transaction-level data into customer-level summaries (Recency, Frequency, Monetary).

## Modeling

K-Means is chosen for this project because:
<ul>
    <li>
        It is efficient and scalable for large datasets like Online Retail (thousands of customers).</li>
    <br>
    <li>
        It produces distinct, non-overlapping clusters, which makes it easy to label customer types (e.g., “high-value,” “occasional,” “inactive”).
    </li>
    <br>
    <li>
        The resulting centroids directly describe average customer behavior (mean Recency, Frequency, Monetary), which is easy for a business to interpret.
    </li>
</ul>

<img src="clustering.png" alt="image of cluster plot">

## Storytelling

## Impact

## Reference

Dataset: https://archive.ics.uci.edu/dataset/352/online+retail

Information: https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/

Coding help: chat gpt