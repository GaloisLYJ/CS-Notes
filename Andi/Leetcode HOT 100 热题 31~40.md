<!-- GFM-TOC -->
* [31. 二叉树的最近公共祖先](#31-二叉树的最近公共祖先)

* [40. 合并两个有序链表](#40-合并两个有序链表)

* [33. 移动零](#33-移动零)

* [34. 把二叉搜索树转换为累加树](#34-把二叉搜索树转换为累加树)

* [35. 不同路径](#35-不同路径)

* [36. 戳气球](#36-戳气球)

* [37. 打家劫舍III](#37-打家劫舍III)

* [38. 编辑距离](#37-编辑距离)

* [39. 颜色分类](#39-颜色分类)

* [40. 路径总和III](#39-路径总和III)

  <!-- GFM-TOC -->


# 31. 二叉树的最近公共祖先

[Leetcode #236 (Medium)](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/15/binarytree.png">

```html
示例 1:

输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
示例 2:

输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
```

**说明:**

- 所有节点的值都是唯一的。

- p、q 为不同节点且均存在于给定的二叉树中。

  

- 递归
- 迭代
```java
class Solution {

    private TreeNode ans;

    public Solution() {
        // Variable to store LCA node.
        this.ans = null;
    }

    private boolean recurseTree(TreeNode currentNode, TreeNode p, TreeNode q) {

        // If reached the end of a branch, return false.
        if (currentNode == null) {
            return false;
        }

        // Left Recursion. If left recursion returns true, set left = 1 else 0
        int left = this.recurseTree(currentNode.left, p, q) ? 1 : 0;

        // Right Recursion
        int right = this.recurseTree(currentNode.right, p, q) ? 1 : 0;

        // If the current node is one of p or q
        int mid = (currentNode == p || currentNode == q) ? 1 : 0;


        // If any two of the flags left, right or mid become True
        if (mid + left + right >= 2) {
            this.ans = currentNode;
        }

        // Return true if any one of the three bool values is True.
        return (mid + left + right > 0);
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // Traverse the tree
        this.recurseTree(root, p, q);
        return this.ans;
    }
}
```

# 32. 合并两个有序链表

[Leetcode #21(Medium)](https://leetcode-cn.com/problems/merge-two-sorted-lists/)
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

```html
示例：

输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

- 递归
- 迭代
```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        else if (l2 == null) {
            return l1;
        }
        else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        }
        else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }

    }
}
```
# 33. 移动零

[Leetcode #283 (Easy)](https://leetcode-cn.com/problems/move-zeroes/)
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
说明:
必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。
```html
示例:

输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
```

- 双指针
```java
class Solution {
    public void moveZeroes(int[] nums) {
        int k = 0;
        for(int i = 0;i < nums.length;i++){
            if(nums[i] != 0){
                swapArrays(nums,i,k++);
            }
        }
    }
    private void swapArrays(int[] nums,int first,int second){
        if(first == second){
            return;
        }
        int temp = nums[first];
        nums[first] = nums[second];
        nums[second] = temp;
    }
}
```

# 34. 把二叉搜索树转化为累加树

[Leetcode #538 (Easy)](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
给定一个二叉搜索树（Binary Search Tree），把它转换成为累加树（Greater Tree)，使得每个节点的值是原来的节点值加上所有大于它的节点值之和。

```html
例如：

输入: 二叉搜索树:
              5
            /   \
           2     13

输出: 转换为累加树:
             18
            /   \
          20     13
```
- 逆中序遍历
```java
class Solution {
    int sum = 0;
    public TreeNode convertBST(TreeNode root) {
       
       backOrder(root);
       return root;
    }
    public void backOrder(TreeNode root) {
        if(root == null) {
            return ;
        }
        backOrder(root.right);
        root.val += sum;
        sum = root.val;
        backOrder(root.left);
    }
```




# 35. 不同路径

[Leetcode #62  (Medium)](https://leetcode-cn.com/problems/unique-paths/)
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

问总共有多少条不同的路径？
<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/robot_maze.png">
例如，上图是一个7 x 3 的网格。有多少可能的路径？
说明：m 和 n 的值均不超过 100。

```html
示例 1:

输入: m = 3, n = 2
输出: 3
解释:
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右
示例 2:

输入: m = 7, n = 3
输出: 28
```

- 排列组合
- 动态规划
```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; i++) dp[0][i] = 1;
        for (int i = 0; i < m; i++) dp[i][0] = 1;
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];  
    }
}
````

# 36. 戳气球

[Leetcode #312 (Difficult))](https://leetcode-cn.com/problems/burst-balloons/)
有 n 个气球，编号为0 到 n-1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

现在要求你戳破所有的气球。每当你戳破一个气球 i 时，你可以获得 nums[left] * nums[i] * nums[right] 个硬币。 这里的 left 和 right 代表和 i 相邻的两个气球的序号。注意当你戳破了气球 i 后，气球 left 和气球 right 就变成了相邻的气球。
求所能获得硬币的最大数量。

说明:
- 你可以假设 nums[-1] = nums[n] = 1，但注意它们不是真实存在的所以并不能被戳破。
- 0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100

```html
示例:

输入: [3,1,5,8]
输出: 167 
解释: nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
     coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167
```

- 动态规划
```java
    public int maxCoins(int[] nums) {
        int [][]d=new int[nums.length+2][nums.length+2];
        int []a=new int [nums.length+2];
        for(int i=1;i<a.length-1;i++){
            a[i]=nums[i-1];
        }
        a[0]=1;
        a[a.length-1]=1;
        for(int i=0;i<a.length;i++){
            d[i][i]=0;
        }
        for(int i=2;i<a.length;i++){
            for(int j=0;j<a.length-i;j++){
                for(int k=j+1;k<j+i;k++){
                    d[j][j+i]=Math.max(d[j][j+i],d[j][k]+d[k][j+i]+a[j]*a[k]*a[j+i]);
                }
            }
        }
        return d[0][a.length-1];

    }
```

# 37. 打家劫舍III

[Leetcode #337 (Medium)](https://leetcode-cn.com/problems/house-robber-iii/)
在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
```html
示例 1:

输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

输出: 7 
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
示例 2:

输入: [3,4,5,1,3,null,1]

           3
    / \
   4   5
  / \   \ 
 1   3   1

输出: 9
解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.
```
- 递归
```java
class Solution {
    public int rob(TreeNode root){
        return doRob(root);
    }
    private int doRob(TreeNode node){
        if (node == null)
            return 0;
        int include = node.val;
        int exclude = doRob(node.left) + doRob(node.right);
        if (node.left != null)
        {
            include += doRob(node.left.left);
            include += doRob(node.left.right);
        }
        if (node.right != null){
            include += doRob(node.right.left);
            include += doRob(node.right.right);
        }
        return Math.max(include,exclude);
    }
}
```
# 38. 编辑距离

[Leetcode #72 (Difficult)](https://leetcode-cn.com/problems/edit-distance/)
给定两个单词 word1 和 word2，计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：
1. 插入一个字符
2. 删除一个字符
3. 替换一个字符
```html
示例 1:

输入: word1 = "horse", word2 = "ros"
输出: 3
解释: 
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
示例 2:

输入: word1 = "intention", word2 = "execution"
输出: 5
解释: 
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

- 动态规划
```java
class Solution {
  public int minDistance(String word1, String word2) {
    int n = word1.length();
    int m = word2.length();

    // if one of the strings is empty
    if (n * m == 0)
      return n + m;

    // array to store the convertion history
    int [][] d = new int[n + 1][m + 1];

    // init boundaries
    for (int i = 0; i < n + 1; i++) {
      d[i][0] = i;
    }
    for (int j = 0; j < m + 1; j++) {
      d[0][j] = j;
    }

    // DP compute 
    for (int i = 1; i < n + 1; i++) {
      for (int j = 1; j < m + 1; j++) {
        int left = d[i - 1][j] + 1;
        int down = d[i][j - 1] + 1;
        int left_down = d[i - 1][j - 1];
        if (word1.charAt(i - 1) != word2.charAt(j - 1))
          left_down += 1;
        d[i][j] = Math.min(left, Math.min(down, left_down));

      }
    }
    return d[n][m];
  }
}
```

# 39. 颜色分类

[Leetcode #75 (Medium)](https://leetcode-cn.com/problems/sort-colors/)
给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

注意:
不能使用代码库中的排序函数来解决这道题。
```html
示例:

输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]
```
进阶：

- 一个直观的解决方案是使用计数排序的两趟扫描算法。
首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。

- 你能想出一个仅使用常数空间的一趟扫描算法吗？

- 荷兰国旗问题
- 双指针扫描
```java
class Solution {
  /*
  荷兰三色旗问题解
  */
  public void sortColors(int[] nums) {
    // 对于所有 idx < i : nums[idx < i] = 0
    // j是当前考虑元素的下标
    int p0 = 0, curr = 0;
    // 对于所有 idx > k : nums[idx > k] = 2
    int p2 = nums.length - 1;

    int tmp;
    while (curr <= p2) {
      if (nums[curr] == 0) {
        // 交换第 p0个和第curr个元素
        // i++，j++
        tmp = nums[p0];
        nums[p0++] = nums[curr];
        nums[curr++] = tmp;
      }
      else if (nums[curr] == 2) {
        // 交换第k个和第curr个元素
        // p2--
        tmp = nums[curr];
        nums[curr] = nums[p2];
        nums[p2--] = tmp;
      }
      else curr++;
    }
  }
}
```

# 40. 路径总和

[Leetcode 437 (Easy)](https://leetcode-cn.com/problems/path-sum-iii/)

给定一个二叉树，它的每个结点都存放着一个整数值。
找出路径和等于给定数值的路径总数。

路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。

```html
示例：

root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

返回 3。和等于 8 的路径有:

1.  5 -> 3
2.  5 -> 2 -> 1
3.  -3 -> 11
```

- 递归

```java
public int pathSum(TreeNode root, int sum) {
    
  if (root==null)
            return 0;
        return pathSum(root.left, sum) + pathSum(root.right, sum) + dfs(root, sum);
    }
    
    public int dfs(TreeNode node, int sum) {
        if (node==null)
            return 0;
        
        int count = 0;
        if (node.val == sum)
            count = 1;
        
        return count + dfs(node.left, sum - node.val) + dfs(node.right, sum - node.val);
    }
```