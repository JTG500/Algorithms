//
//  main.cpp
//  Algorithm Code
//
//  Created by Eric Foster on 11/8/16.
//  Copyright © 2016 Eric Foster. All rights reserved.
//

#include <iostream>
#include <string>

using namespace std;
/* C program for Merge Sort */
#include<stdlib.h>
#include<stdio.h>
#define MAX_TREE_HT 100

// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
/* MERGE SORT STARTS HERE ********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/
void merge(int arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
    
    /* create temp arrays */
    int L[n1], R[n2];
    
    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];
    
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    
    /* Copy the remaining elements of L[], if there
     are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }
    
    /* Copy the remaining elements of R[], if there
     are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

/* l is for left index and r is right index of the
 sub-array of arr to be sorted */
void mergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for
        // large l and h
        int m = l+(r-l)/2;
        
        // Sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);
        
        merge(arr, l, m, r);
    }
}

/* UTILITY FUNCTIONS */
/* Function to print an array */
void printMerge(int A[], int size)
{
    int i;
    for (i=0; i < size; i++)
        printf("%d ", A[i]);
    printf("\n");
}

/* THIS IS ALL FOR MERGE SORT CODE **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/




/* INSERTION SORT STARTS HERE ********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/


/* Function to sort an array using insertion sort*/
void insertionSort(int arr[], int n)
{
    int i, key, j;
    for (i = 1; i < n; i++)
    {
        key = arr[i];
        j = i-1;
        
        /* Move elements of arr[0..i-1], that are
         greater than key, to one position ahead
         of their current position */
        while (j >= 0 && arr[j] > key)
        {
            arr[j+1] = arr[j];
            j = j-1;
        }
        arr[j+1] = key;
    }
}

// A utility function ot print an array of size n
void printInsertion(int arr[], int n)
{
    int i;
    for (i=0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

/* THIS IS ALL FOR INSERTION SORT CODE **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/




/* THIS IS ALL FOR RADIX SORT CODE **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/




// A utility function to get maximum value in arr[]
int getMax(int arr[], int n)
{
    int mx = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > mx)
            mx = arr[i];
    return mx;
}

// A function to do counting sort of arr[] according to
// the digit represented by exp.
void countSort(int arr[], int n, int exp)
{
    int output[n]; // output array
    int i, count[10] = {0};
    
    // Store count of occurrences in count[]
    for (i = 0; i < n; i++)
        count[ (arr[i]/exp)%10 ]++;
    
    // Change count[i] so that count[i] now contains actual
    //  position of this digit in output[]
    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];
    
    // Build the output array
    for (i = n - 1; i >= 0; i--)
    {
        output[count[ (arr[i]/exp)%10 ] - 1] = arr[i];
        count[ (arr[i]/exp)%10 ]--;
    }
    
    // Copy the output array to arr[], so that arr[] now
    // contains sorted numbers according to current digit
    for (i = 0; i < n; i++)
        arr[i] = output[i];
}

// The main function to that sorts arr[] of size n using
// Radix Sort
void radixsort(int arr[], int n)
{
    // Find the maximum number to know number of digits
    int m = getMax(arr, n);
    
    // Do counting sort for every digit. Note that instead
    // of passing digit number, exp is passed. exp is 10^i
    // where i is current digit number
    for (int exp = 1; m/exp > 0; exp *= 10)
        countSort(arr, n, exp);
}

// A utility function to print an array
void printRadix(int arr[], int n)
{
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
}

/* THIS IS ALL FOR RADIX SORT CODE **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/


/* THIS IS ALL FOR QUICKSORT CODE **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/


void quickSwap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

/* This function takes last element as pivot, places
 the pivot element at its correct position in sorted
 array, and places all smaller (smaller than pivot)
 to left of pivot and all greater elements to right
 of pivot */
int partition (int arr[], int low, int high)
{
    int pivot = arr[high];    // pivot
    int i = (low - 1);  // Index of smaller element
    
    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot)
        {
            i++;    // increment index of smaller element
            quickSwap(&arr[i], &arr[j]);
        }
    }
    quickSwap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

/* The main function that implements QuickSort
 arr[] --> Array to be sorted,
 low  --> Starting index,
 high  --> Ending index */
void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
         at right place */
        int pi = partition(arr, low, high);
        
        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

/* Function to print an array */
void printQuick(int arr[], int size)
{
    int i;
    for (i=0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

/* THIS IS ALL FOR QUICKSORT CODE **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/

/* THIS IS ALL FOR BUBBLE SORT CODE **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/

void bubbleSwap(int *xp, int *yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

// A function to implement bubble sort
void bubbleSort(int arr[], int n)
{
    int i, j;
    for (i = 0; i < n-1; i++)
        
        // Last i elements are already in place
        for (j = 0; j < n-i-1; j++)
            if (arr[j] > arr[j+1])
                bubbleSwap(&arr[j], &arr[j+1]);
}

/* Function to print an array */
void printBubble(int arr[], int size)
{
    for (int i=0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

/* THIS IS ALL FOR BUBBLE SORT CODE **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/

/* THIS IS ALL FOR HEAP SORT CODE **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/

void heapify(int arr[], int n, int i)
{
    int largest = i;  // Initialize largest as root
    int l = 2*i + 1;  // left = 2*i + 1
    int r = 2*i + 2;  // right = 2*i + 2
    
    // If left child is larger than root
    if (l < n && arr[l] > arr[largest])
        largest = l;
    
    // If right child is larger than largest so far
    if (r < n && arr[r] > arr[largest])
        largest = r;
    
    // If largest is not root
    if (largest != i)
    {
        swap(arr[i], arr[largest]);
        
        // Recursively heapify the affected sub-tree
        heapify(arr, n, largest);
    }
}

// main function to do heap sort
void heapSort(int arr[], int n)
{
    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
    
    // One by one extract an element from heap
    for (int i=n-1; i>=0; i--)
    {
        // Move current root to end
        swap(arr[0], arr[i]);
        
        // call max heapify on the reduced heap
        heapify(arr, i, 0);
    }
}

/* A utility function to print array of size n */
void printHeap(int arr[], int n)
{
    for (int i=0; i<n; ++i)
        cout << arr[i] << " ";
    cout << "\n";
}

/* THIS IS ALL FOR HEAP SORT CODE **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/

/* THIS IS ALL FOR GREEDY CODE (ACTIVITY SELECTION PROBLEM) **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/


// Prints a maximum set of activities that can be done by a single
// person, one at a time.
//  n   -->  Total number of activities
//  s[] -->  An array that contains start time of all activities
//  f[] -->  An array that contains finish time of all activities
void printMaxActivities(int s[], int f[], int n)
{
    int i, j;
    
    printf ("Following activities are selected \n");
    
    // The first activity always gets selected
    i = 0;
    printf("%d ", i);
    
    // Consider rest of the activities
    for (j = 1; j < n; j++)
    {
        // If this activity has start time greater than or
        // equal to the finish time of previously selected
        // activity, then select it
        if (s[j] >= f[i])
        {
            printf ("%d ", j);
            i = j;
        }
    }
}

/* THIS IS ALL FOR GREEDY CODE (KRUSKALS SPAN TREE PROBLEM) **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/



// a structure to represent a weighted edge in graph
struct Edge
{
    int src, dest, weight;
};

// a structure to represent a connected, undirected and weighted graph
struct Graph
{
    // V-> Number of vertices, E-> Number of edges
    int V, E;
    
    // graph is represented as an array of edges. Since the graph is
    // undirected, the edge from src to dest is also edge from dest
    // to src. Both are counted as 1 edge here.
    struct Edge* edge;
};

// Creates a graph with V vertices and E edges
struct Graph* createGraph(int V, int E)
{
    struct Graph* graph = (struct Graph*) malloc( sizeof(struct Graph) );
    graph->V = V;
    graph->E = E;
    
    graph->edge = (struct Edge*) malloc( graph->E * sizeof( struct Edge ) );
    
    return graph;
}

// A structure to represent a subset for union-find
struct subset
{
    int parent;
    int rank;
};

// A utility function to find set of an element i
// (uses path compression technique)
int find(struct subset subsets[], int i)
{
    // find root and make root as parent of i (path compression)
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
    
    return subsets[i].parent;
}

// A function that does union of two sets of x and y
// (uses union by rank)
void Union(struct subset subsets[], int x, int y)
{
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);
    
    // Attach smaller rank tree under root of high rank tree
    // (Union by Rank)
    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;
    
    // If ranks are same, then make one as root and increment
    // its rank by one
    else
    {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
}

// Compare two edges according to their weights.
// Used in qsort() for sorting an array of edges
int myComp(const void* a, const void* b)
{
    struct Edge* a1 = (struct Edge*)a;
    struct Edge* b1 = (struct Edge*)b;
    return a1->weight > b1->weight;
}

// The main function to construct MST using Kruskal's algorithm
void KruskalMST(struct Graph* graph)
{
    int V = graph->V;
    struct Edge result[V];  // Tnis will store the resultant MST
    int e = 0;  // An index variable, used for result[]
    int i = 0;  // An index variable, used for sorted edges
    
    // Step 1:  Sort all the edges in non-decreasing order of their weight
    // If we are not allowed to change the given graph, we can create a copy of
    // array of edges
    qsort(graph->edge, graph->E, sizeof(graph->edge[0]), myComp);
    
    // Allocate memory for creating V ssubsets
    struct subset *subsets =
    (struct subset*) malloc( V * sizeof(struct subset) );
    
    // Create V subsets with single elements
    for (int v = 0; v < V; ++v)
    {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }
    
    // Number of edges to be taken is equal to V-1
    while (e < V - 1)
    {
        // Step 2: Pick the smallest edge. And increment the index
        // for next iteration
        struct Edge next_edge = graph->edge[i++];
        
        int x = find(subsets, next_edge.src);
        int y = find(subsets, next_edge.dest);
        
        // If including this edge does't cause cycle, include it
        // in result and increment the index of result for next edge
        if (x != y)
        {
            result[e++] = next_edge;
            Union(subsets, x, y);
        }
        // Else discard the next_edge
    }
    
    // print the contents of result[] to display the built MST
    printf("Following are the edges in the constructed MST\n");
    for (i = 0; i < e; ++i)
        printf("%d -- %d == %d\n", result[i].src, result[i].dest,
               result[i].weight);
    return;
}

/* THIS IS ALL FOR GREEDY CODE (HUFFMAN PROBLEM) **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/


struct MinHeapNode
{
    char data;  // One of the input characters
    unsigned freq;  // Frequency of the character
    struct MinHeapNode *left, *right; // Left and right child of this node
};

// A Min Heap:  Collection of min heap (or Hufmman tree) nodes
struct MinHeap
{
    unsigned size;    // Current size of min heap
    unsigned capacity;   // capacity of min heap
    struct MinHeapNode **array;  // Attay of minheap node pointers
};

// A utility function allocate a new min heap node with given character
// and frequency of the character
struct MinHeapNode* newNode(char data, unsigned freq)
{
    struct MinHeapNode* temp =
    (struct MinHeapNode*) malloc(sizeof(struct MinHeapNode));
    temp->left = temp->right = NULL;
    temp->data = data;
    temp->freq = freq;
    return temp;
}

// A utility function to create a min heap of given capacity
struct MinHeap* createMinHeap(unsigned capacity)
{
    struct MinHeap* minHeap =
    (struct MinHeap*) malloc(sizeof(struct MinHeap));
    minHeap->size = 0;  // current size is 0
    minHeap->capacity = capacity;
    minHeap->array =
    (struct MinHeapNode**)malloc(minHeap->capacity * sizeof(struct MinHeapNode*));
    return minHeap;
}

// A utility function to swap two min heap nodes
void swapMinHeapNode(struct MinHeapNode** a, struct MinHeapNode** b)
{
    struct MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}

// The standard minHeapify function.
void minHeapify(struct MinHeap* minHeap, int idx)
{
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    
    if (left < minHeap->size &&
        minHeap->array[left]->freq < minHeap->array[smallest]->freq)
        smallest = left;
    
    if (right < minHeap->size &&
        minHeap->array[right]->freq < minHeap->array[smallest]->freq)
        smallest = right;
    
    if (smallest != idx)
    {
        swapMinHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);
        minHeapify(minHeap, smallest);
    }
}

// A utility function to check if size of heap is 1 or not
int isSizeOne(struct MinHeap* minHeap)
{
    return (minHeap->size == 1);
}

// A standard function to extract minimum value node from heap
struct MinHeapNode* extractMin(struct MinHeap* minHeap)
{
    struct MinHeapNode* temp = minHeap->array[0];
    minHeap->array[0] = minHeap->array[minHeap->size - 1];
    --minHeap->size;
    minHeapify(minHeap, 0);
    return temp;
}

// A utility function to insert a new node to Min Heap
void insertMinHeap(struct MinHeap* minHeap, struct MinHeapNode* minHeapNode)
{
    ++minHeap->size;
    int i = minHeap->size - 1;
    while (i && minHeapNode->freq < minHeap->array[(i - 1)/2]->freq)
    {
        minHeap->array[i] = minHeap->array[(i - 1)/2];
        i = (i - 1)/2;
    }
    minHeap->array[i] = minHeapNode;
}

// A standard funvtion to build min heap
void buildMinHeap(struct MinHeap* minHeap)
{
    int n = minHeap->size - 1;
    int i;
    for (i = (n - 1) / 2; i >= 0; --i)
        minHeapify(minHeap, i);
}

// A utility function to print an array of size n
void printArr(int arr[], int n)
{
    int i;
    for (i = 0; i < n; ++i)
        printf("%d", arr[i]);
    printf("\n");
}

// Utility function to check if this node is leaf
int isLeaf(struct MinHeapNode* root)
{
    return !(root->left) && !(root->right) ;
}

// Creates a min heap of capacity equal to size and inserts all character of
// data[] in min heap. Initially size of min heap is equal to capacity
struct MinHeap* createAndBuildMinHeap(char data[], int freq[], int size)
{
    struct MinHeap* minHeap = createMinHeap(size);
    for (int i = 0; i < size; ++i)
        minHeap->array[i] = newNode(data[i], freq[i]);
    minHeap->size = size;
    buildMinHeap(minHeap);
    return minHeap;
}

// The main function that builds Huffman tree
struct MinHeapNode* buildHuffmanTree(char data[], int freq[], int size)
{
    struct MinHeapNode *left, *right, *top;
    
    // Step 1: Create a min heap of capacity equal to size.  Initially, there are
    // modes equal to size.
    struct MinHeap* minHeap = createAndBuildMinHeap(data, freq, size);
    
    // Iterate while size of heap doesn't become 1
    while (!isSizeOne(minHeap))
    {
        // Step 2: Extract the two minimum freq items from min heap
        left = extractMin(minHeap);
        right = extractMin(minHeap);
        
        // Step 3:  Create a new internal node with frequency equal to the
        // sum of the two nodes frequencies. Make the two extracted node as
        // left and right children of this new node. Add this node to the min heap
        // '$' is a special value for internal nodes, not used
        top = newNode('$', left->freq + right->freq);
        top->left = left;
        top->right = right;
        insertMinHeap(minHeap, top);
    }
    
    // Step 4: The remaining node is the root node and the tree is complete.
    return extractMin(minHeap);
}

// Prints huffman codes from the root of Huffman Tree.  It uses arr[] to
// store codes
void printCodes(struct MinHeapNode* root, int arr[], int top)
{
    // Assign 0 to left edge and recur
    if (root->left)
    {
        arr[top] = 0;
        printCodes(root->left, arr, top + 1);
    }
    
    // Assign 1 to right edge and recur
    if (root->right)
    {
        arr[top] = 1;
        printCodes(root->right, arr, top + 1);
    }
    
    // If this is a leaf node, then it contains one of the input
    // characters, print the character and its code from arr[]
    if (isLeaf(root))
    {
        printf("%c: ", root->data);
        printArr(arr, top);
    }
}

// The main function that builds a Huffman Tree and print codes by traversing
// the built Huffman Tree
void HuffmanCodes(char data[], int freq[], int size)
{
    //  Construct Huffman Tree
    struct MinHeapNode* root = buildHuffmanTree(data, freq, size);
    
    // Print Huffman codes using the Huffman tree built above
    int arr[MAX_TREE_HT], top = 0;
    printCodes(root, arr, top);
}

/* THIS IS ALL FOR GREEDY CODE (FRACTIONAL KNAPSACK PROBLEM) **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/

// Stucture for Item which store weight and corresponding
// value of Item
struct Item
{
    int value, weight;
    
    // Constructor
    Item(int value, int weight) : value(value), weight(weight)
    {}
};

// Comparison function to sort Item according to val/weight ratio
bool cmp(struct Item a, struct Item b)
{
    double r1 = (double)a.value / a.weight;
    double r2 = (double)b.value / b.weight;
    return r1 > r2;
}

// Main greedy function to solve problem
double fractionalKnapsack(int W, struct Item arr[], int n)
{
    //    sorting Item on basis of ration
    sort(arr, arr + n, cmp);
    
    //    Uncomment to see new order of Items with their ratio
    
     for (int i = 0; i < n; i++)
     {
     cout << arr[i].value << "  " << arr[i].weight << " : "
     << ((double)arr[i].value / arr[i].weight) << endl;
     }
    
    
    int curWeight = 0;  // Current weight in knapsack
    double finalvalue = 0.0; // Result (value in Knapsack)
    
    // Looping through all Items
    for (int i = 0; i < n; i++)
    {
        // If adding Item won't overflow, add it completely
        if (curWeight + arr[i].weight <= W)
        {
            curWeight += arr[i].weight;
            finalvalue += arr[i].value;
        }
        
        // If we can't add current Item, add fractional part of it
        else
        {
            int remain = W - curWeight;
            finalvalue += arr[i].value * ((double) remain / arr[i].weight);
            break;
        }
    }
    
    // Returning final value
    return finalvalue;
}




/* THIS IS ALL FOR DYNAMIC PROGRAMMING (0-1 KNAPSACK) **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/

int maxKnap(int a, int b) { return (a > b)? a : b; }

// Returns the maximum value that can be put in a knapsack of capacity W
int knapSack(int W, int wt[], int val[], int n)
{
    // Base Case
    if (n == 0 || W == 0)
        return 0;
    
    // If weight of the nth item is more than Knapsack capacity W, then
    // this item cannot be included in the optimal solution
    if (wt[n-1] > W)
        return knapSack(W, wt, val, n-1);
    
    // Return the maximum of two cases:
    // (1) nth item included
    // (2) not included
    else return maxKnap( val[n-1] + knapSack(W-wt[n-1], wt, val, n-1),
                    knapSack(W, wt, val, n-1)
                    );
}

/* THIS IS ALL FOR DYNAMIC PROGRAMMING (ROD CUTTING) **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/



int maxRod(int a, int b) { return (a > b)? a : b;}

/* Returns the best obtainable price for a rod of length n and
 price[] as prices of different pieces */
int cutRod(int price[], int n)
{
    if (n <= 0)
        return 0;
    int max_val = INT_MIN;
    
    // Recursively cut the rod in different pieces and compare different
    // configurations
    for (int i = 0; i<n; i++)
    {
        max_val = maxRod(max_val, price[i] + cutRod(price, n-i-1));
        cout << max_val << "\n" << endl;
    }
    
    return max_val;
}

/* THIS IS ALL FOR DYNAMIC PROGRAMMING (COIN COUNTING) **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/

// Returns the count of ways we can sum  S[0...m-1] coins to get sum n
int count( int S[], int m, int n )
{
    int i, j, x, y;
    
    // We need n+1 rows as the table is consturcted in bottom up manner using
    // the base case 0 value case (n = 0)
    int table[n+1][m];
    
    // Fill the enteries for 0 value case (n = 0)
    for (i=0; i<m; i++)
        table[0][i] = 1;
    
    // Fill rest of the table enteries in bottom up manner
    for (i = 1; i < n+1; i++)
    {
        for (j = 0; j < m; j++)
        {
            // Count of solutions including S[j]
            x = (i-S[j] >= 0)? table[i - S[j]][j]: 0;
            
            // Count of solutions excluding S[j]
            y = (j >= 1)? table[i][j-1]: 0;
            
            // total count
            table[i][j] = x + y;
            cout << table[i][j];
        }
        cout << endl;
    }
    return table[n][m-1];
}

/* THIS IS ALL FOR RED BLACK TREES  **************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/



struct node
{
    int key;
    node *parent;
    char color;
    node *left;
    node *right;
};
class RBtree
{
    node *root;
    node *q;
    public :
    RBtree()
    {
        q=NULL;
        root=NULL;
    }
    void insert();
    void insertfix(node *);
    void leftrotate(node *);
    void rightrotate(node *);
    void del();
    node* successor(node *);
    void delfix(node *);
    void disp();
    void display( node *);
    void search();
};
void RBtree::insert()
{
    int z,i=0;
    cout<<"\nEnter key of the node to be inserted: ";
    cin>>z;
    node *p,*q;
    node *t=new node;
    t->key=z;
    t->left=NULL;
    t->right=NULL;
    t->color='r';
    p=root;
    q=NULL;
    if(root==NULL)
    {
        root=t;
        t->parent=NULL;
    }
    else
    {
        while(p!=NULL)
        {
            q=p;
            if(p->key<t->key)
                p=p->right;
            else
                p=p->left;
        }
        t->parent=q;
        if(q->key<t->key)
            q->right=t;
        else
            q->left=t;
    }
    insertfix(t);
}
void RBtree::insertfix(node *t)
{
    node *u;
    if(root==t)
    {
        t->color='b';
        return;
    }
    while(t->parent!=NULL&&t->parent->color=='r')
    {
        node *g=t->parent->parent;
        if(g->left==t->parent)
        {
            if(g->right!=NULL)
            {
                u=g->right;
                if(u->color=='r')
                {
                    t->parent->color='b';
                    u->color='b';
                    g->color='r';
                    t=g;
                }
            }
            else
            {
                if(t->parent->right==t)
                {
                    t=t->parent;
                    leftrotate(t);
                }
                t->parent->color='b';
                g->color='r';
                rightrotate(g);
            }
        }
        else
        {
            if(g->left!=NULL)
            {
                u=g->left;
                if(u->color=='r')
                {
                    t->parent->color='b';
                    u->color='b';
                    g->color='r';
                    t=g;
                }
            }
            else
            {
                if(t->parent->left==t)
                {
                    t=t->parent;
                    rightrotate(t);
                }
                t->parent->color='b';
                g->color='r';
                leftrotate(g);
            }
        }
        root->color='b';
    }
}

void RBtree::del()
{
    if(root==NULL)
    {
        cout<<"\nEmpty Tree." ;
        return ;
    }
    int x;
    cout<<"\nEnter the key of the node to be deleted: ";
    cin>>x;
    node *p;
    p=root;
    node *y=NULL;
    node *q=NULL;
    int found=0;
    while(p!=NULL&&found==0)
    {
        if(p->key==x)
            found=1;
        if(found==0)
        {
            if(p->key<x)
                p=p->right;
            else
                p=p->left;
        }
    }
    if(found==0)
    {
        cout<<"\nElement Not Found.";
        return ;
    }
    else
    {
        cout<<"\nDeleted Element: "<<p->key;
        cout<<"\nColour: ";
        if(p->color=='b')
            cout<<"Black\n";
        else
            cout<<"Red\n";
        
        if(p->parent!=NULL)
            cout<<"\nParent: "<<p->parent->key;
        else
            cout<<"\nThere is no parent of the node.  ";
        if(p->right!=NULL)
            cout<<"\nRight Child: "<<p->right->key;
        else
            cout<<"\nThere is no right child of the node.  ";
        if(p->left!=NULL)
            cout<<"\nLeft Child: "<<p->left->key;
        else
            cout<<"\nThere is no left child of the node.  ";
        cout<<"\nNode Deleted.";
        if(p->left==NULL||p->right==NULL)
            y=p;
        else
            y=successor(p);
        if(y->left!=NULL)
            q=y->left;
        else
        {
            if(y->right!=NULL)
                q=y->right;
            else
                q=NULL;
        }
        if(q!=NULL)
            q->parent=y->parent;
        if(y->parent==NULL)
            root=q;
        else
        {
            if(y==y->parent->left)
                y->parent->left=q;
            else
                y->parent->right=q;
        }
        if(y!=p)
        {
            p->color=y->color;
            p->key=y->key;
        }
        if(y->color=='b')
            delfix(q);
    }
}

void RBtree::delfix(node *p)
{
    node *s;
    while(p!=root&&p->color=='b')
    {
        if(p->parent->left==p)
        {
            s=p->parent->right;
            if(s->color=='r')
            {
                s->color='b';
                p->parent->color='r';
                leftrotate(p->parent);
                s=p->parent->right;
            }
            if(s->right->color=='b'&&s->left->color=='b')
            {
                s->color='r';
                p=p->parent;
            }
            else
            {
                if(s->right->color=='b')
                {
                    s->left->color=='b';
                    s->color='r';
                    rightrotate(s);
                    s=p->parent->right;
                }
                s->color=p->parent->color;
                p->parent->color='b';
                s->right->color='b';
                leftrotate(p->parent);
                p=root;
            }
        }
        else
        {
            s=p->parent->left;
            if(s->color=='r')
            {
                s->color='b';
                p->parent->color='r';
                rightrotate(p->parent);
                s=p->parent->left;
            }
            if(s->left->color=='b'&&s->right->color=='b')
            {
                s->color='r';
                p=p->parent;
            }
            else
            {
                if(s->left->color=='b')
                {
                    s->right->color='b';
                    s->color='r';
                    leftrotate(s);
                    s=p->parent->left;
                }
                s->color=p->parent->color;
                p->parent->color='b';
                s->left->color='b';
                rightrotate(p->parent);
                p=root;
            }
        }
        p->color='b';
        root->color='b';
    }
}

void RBtree::leftrotate(node *p)
{
    if(p->right==NULL)
        return ;
    else
    {
        node *y=p->right;
        if(y->left!=NULL)
        {
            p->right=y->left;
            y->left->parent=p;
        }
        else
            p->right=NULL;
        if(p->parent!=NULL)
            y->parent=p->parent;
        if(p->parent==NULL)
            root=y;
        else
        {
            if(p==p->parent->left)
                p->parent->left=y;
            else
                p->parent->right=y;
        }
        y->left=p;
        p->parent=y;
    }
}
void RBtree::rightrotate(node *p)
{
    if(p->left==NULL)
        return ;
    else
    {
        node *y=p->left;
        if(y->right!=NULL)
        {
            p->left=y->right;
            y->right->parent=p;
        }
        else
            p->left=NULL;
        if(p->parent!=NULL)
            y->parent=p->parent;
        if(p->parent==NULL)
            root=y;
        else
        {
            if(p==p->parent->left)
                p->parent->left=y;
            else
                p->parent->right=y;
        }
        y->right=p;
        p->parent=y;
    }
}

node* RBtree::successor(node *p)
{
    node *y=NULL;
    if(p->left!=NULL)
    {
        y=p->left;
        while(y->right!=NULL)
            y=y->right;
    }
    else
    {
        y=p->right;
        while(y->left!=NULL)
            y=y->left;
    }
    return y;
}

void RBtree::disp()
{
    display(root);
}
void RBtree::display(node *p)
{
    if(root==NULL)
    {
        cout<<"\nEmpty Tree.";
        return ;
    }
    if(p!=NULL)
    {
        cout<<"\n\t NODE: ";
        cout<<"\n Key: "<<p->key;
        cout<<"\n Colour: ";
        if(p->color=='b')
            cout<<"Black";
        else
            cout<<"Red";
        if(p->parent!=NULL)
            cout<<"\n Parent: "<<p->parent->key;
        else
            cout<<"\n There is no parent of the node.  ";
        if(p->right!=NULL)
            cout<<"\n Right Child: "<<p->right->key;
        else
            cout<<"\n There is no right child of the node.  ";
        if(p->left!=NULL)
            cout<<"\n Left Child: "<<p->left->key;
        else
            cout<<"\n There is no left child of the node.  ";
        cout<<endl;
        if(p->left)
        {
            cout<<"\n\nLeft:\n";
            display(p->left);
        }
        /*else
         cout<<"\nNo Left Child.\n";*/
        if(p->right)
        {
            cout<<"\n\nRight:\n";
            display(p->right);
        }
        /*else
         cout<<"\nNo Right Child.\n"*/
    }
}
void RBtree::search()
{
    if(root==NULL)
    {
        cout<<"\nEmpty Tree\n" ;
        return  ;
    }
    int x;
    cout<<"\n Enter key of the node to be searched: ";
    cin>>x;
    node *p=root;
    int found=0;
    while(p!=NULL&& found==0)
    {
        if(p->key==x)
            found=1;
        if(found==0)
        {
            if(p->key<x)
                p=p->right;
            else
                p=p->left;
        }
    }
    if(found==0)
        cout<<"\nElement Not Found.";
    else
    {
        cout<<"\n\t FOUND NODE: ";
        cout<<"\n Key: "<<p->key;
        cout<<"\n Colour: ";
        if(p->color=='b')
            cout<<"Black";
        else
            cout<<"Red";
        if(p->parent!=NULL)
            cout<<"\n Parent: "<<p->parent->key;
        else
            cout<<"\n There is no parent of the node.  ";
        if(p->right!=NULL)
            cout<<"\n Right Child: "<<p->right->key;
        else
            cout<<"\n There is no right child of the node.  ";
        if(p->left!=NULL)
            cout<<"\n Left Child: "<<p->left->key;
        else
            cout<<"\n There is no left child of the node.  ";
        cout<<endl;
        
    }
}



int main() {
    
    string choice, greedyChoice, dynamicChoice;
    bool quit = false;
    do {
    cout << "Please choose a sorting algorithm: ";
    getline(cin, choice);
    cout << "\n\n" << endl;
    
    if (choice.compare("merge sort") == 0)
    {
            int arr[] = {12, 11, 13, 5, 6, 7}; // the elements to sort
            int arr_size = sizeof(arr)/sizeof(arr[0]);
        
            printf("Given array is \n");
            printMerge(arr, arr_size);
        
            mergeSort(arr, 0, arr_size - 1);
        
            printf("\nSorted array is \n");
            printMerge(arr, arr_size);
    }
    else if (choice.compare("insertion sort") == 0)
    {
        int arr[] = {12, 11, 13, 5, 6}; // the elements to sort
        int n = sizeof(arr)/sizeof(arr[0]);
        
        insertionSort(arr, n);
        printInsertion(arr, n);
    }
    else if (choice.compare("radix sort") == 0)
    {
        int arr[] = {170, 45, 75, 90, 802, 24, 2, 66}; // elements to sort
        int n = sizeof(arr)/sizeof(arr[0]);
        radixsort(arr, n);
        printRadix(arr, n);
    }
    else if (choice.compare("quicksort") == 0)
    {
        int arr[] = {10, 7, 8, 9, 1, 5}; // elements to sort
        int n = sizeof(arr)/sizeof(arr[0]);
        quickSort(arr, 0, n-1);
        printf("Sorted array: \n");
        printQuick(arr, n);
    }
    else if (choice.compare("bubble sort") == 0)
    {
        int arr[] = {64, 34, 25, 12, 22, 11, 90}; // elements to sort
        int n = sizeof(arr)/sizeof(arr[0]);
        bubbleSort(arr, n);
        printf("Sorted array: \n");
        printBubble(arr, n);
    }
    else if (choice.compare("heap sort") == 0)
    {
        int arr[] = {18, 31, 4, 17, 22, 6, -5, 17, 9, 30}; // elements to sort
        int n = sizeof(arr)/sizeof(arr[0]);
        
        heapSort(arr, n);
        
        cout << "Sorted array is \n";
        printHeap(arr, n);
    }
    else if (choice.compare("greedy") == 0)
    {
        cout << "Please choose a greedy algorithm: activity selection, min span tree, huffman, fractional knapsack" << endl;
        
        getline(cin, greedyChoice);
        if (greedyChoice.compare("activity selection") == 0)
        {
            int s[] =  {1, 3, 0, 5, 8, 5};
            int f[] =  {2, 4, 6, 7, 9, 9};
            int n = sizeof(s)/sizeof(s[0]);
            printMaxActivities(s, f, n);
            getchar();
        }
        else if (greedyChoice.compare("min span tree") == 0)
        {
            /* Let us create following weighted graph
             10
             0--------1
             |  \     |
             6|   5\   |15
             |      \ |
             2--------3
             4       */
            int V = 4;  // Number of vertices in graph
            int E = 5;  // Number of edges in graph
            struct Graph* graph = createGraph(V, E);
            
            
            // add edge 0-1
            graph->edge[0].src = 0;
            graph->edge[0].dest = 1;
            graph->edge[0].weight = 10;
            
            // add edge 0-2
            graph->edge[1].src = 0;
            graph->edge[1].dest = 2;
            graph->edge[1].weight = 6;
            
            // add edge 0-3
            graph->edge[2].src = 0;
            graph->edge[2].dest = 3;
            graph->edge[2].weight = 5;
            
            // add edge 1-3
            graph->edge[3].src = 1;
            graph->edge[3].dest = 3;
            graph->edge[3].weight = 15;
            
            // add edge 2-3
            graph->edge[4].src = 2;
            graph->edge[4].dest = 3;
            graph->edge[4].weight = 4;
            
            KruskalMST(graph);
            
            
        }
        else if (greedyChoice.compare("huffman") == 0)
        {
            char arr[] = {'a', 'b', 'c', 'd', 'e', 'f'}; // elements
            int freq[] = {5, 9, 12, 13, 16, 45}; // frequencies
            int size = sizeof(arr)/sizeof(arr[0]);
            HuffmanCodes(arr, freq, size);
        }
        else if (greedyChoice.compare("fractional knapsack") == 0)
        {
            int W = 50;   //    Weight of knapsack
            Item arr[] = {{227, 1}, {175, 1}}; // {value, weight}
            
            int n = sizeof(arr) / sizeof(arr[0]);
            
            cout << "Maximum value we can obtain = "
            << fractionalKnapsack(W, arr, n);        }
        else if (greedyChoice.compare("quit") == 0)
        {
            choice = "quit";
        }
        else
            cout << "Invalid choice, check spelling" << endl;
    }
    else if (choice.compare("dynamic") == 0)
    {
        cout << "Please choose which problem: 0-1 knapsack, rod cutting, coin counting" << endl;
        getline(cin, dynamicChoice);
        
        if (dynamicChoice.compare("0-1 knapsack") == 0)
        {
            int val[] = {60, 100, 120};
            int wt[] = {35, 20, 27};
            int  W = 50;
            int n = sizeof(val)/sizeof(val[0]);
            printf("%d\n", knapSack(W, wt, val, n));
        }
        else if (dynamicChoice.compare("rod cutting") == 0)
        {
            int arr[] = {1, 5, 8, 9, 10, 17, 17, 20}; // value of each piece
            int size = sizeof(arr)/sizeof(arr[0]);
            printf("Maximum Obtainable Value is %d\n", cutRod(arr, size));
            getchar();
        }
        else if (dynamicChoice.compare("coin counting") == 0)
        {
            int arr[] = {1, 2, 3}; // value of coins
            int m = sizeof(arr)/sizeof(arr[0]);
            int n = 10; // number to answer Total Value
            printf(" %d ", count(arr, m, n));
        }
        else if (dynamicChoice.compare("quit") == 0)
        {
            choice = "quit";
        }
        else
            cout << "Invalid response, check spelling" << endl;
        
    }
    else if (choice.compare("red black trees") == 0)
    {
        int ch,y=0;
        RBtree obj;
        do
        {
            cout<<"\n\t RED BLACK TREE " ;
            cout<<"\n 1. Insert in the tree ";
            cout<<"\n 2. Delete a node from the tree";
            cout<<"\n 3. Search for an element in the tree";
            cout<<"\n 4. Display the tree ";
            cout<<"\n 5. Exit " ;
            cout<<"\nEnter Your Choice: ";
            cin>>ch;
            switch(ch)
            {
                case 1 : obj.insert();
                    cout<<"\nNode Inserted.\n";
                    break;
                case 2 : obj.del();
                    break;
                case 3 : obj.search();
                    break;
                case 4 : obj.disp();
                    break;
                case 5 : y=1;
                    break;
                default : cout<<"\nEnter a Valid Choice.";
            }
            cout<<endl;
            
        }while(y!=1);
    }
    else if (choice.compare("list") == 0)
    {
        cout << "list of algorithms on file:\nmerge sort\ninsertion sort\nradix sort\nquicksort\nbubble sort\nheap sort\ngreedy\ndynamic\nred black trees" << endl;
    }
        
    else if (choice.compare("quit") == 0)
    {
        quit = true;
    }
        
    else
    {
        cout << "Invalid choice, Please choose one of the following:\nmerge sort\ninsertion sort\nradix sort\nquicksort\nbubble sort\nheap sort\ngreedy\ndynamic\nred black trees" << endl;
    }
    }
    while (quit != true);
    return 0;
}
