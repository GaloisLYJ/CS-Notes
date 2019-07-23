<!-- GFM-TOC -->
* [21. 求众数](#21-求众数)
* [22. 实现Trie(前缀树)](#22-实现Trie(前缀树))
* [23. 从前序与中序遍历序列构造二叉树](#23-从前序与中序遍历序列构造二叉树)
* [24. 每日温度](#24-每日温度)
* [25. 数组中的第K个最大元素](#25-数组中的第K个最大元素)
* [26. 二叉树的层次遍历](#26-二叉树的层次遍历)
* [27. 前K个高频元素](#27-前K个高频元素)
* [28. 字母异位词分组](#28-字母异位词分组)
* [29. 回文子串](#29-回文子串)
* [30. 盛最多水的容器](#30-盛最多水的容器)
  <!-- GFM-TOC -->


# 21. 求众数

[Leetcode #169 (Easy)](https://leetcode-cn.com/problems/majority-element/)

给定一个大小为 *n* 的数组，找到其中的众数。众数是指在数组中出现次数**大于** `⌊ n/2 ⌋` 的元素。
你可以假设数组是非空的，并且给定的数组总是存在众数。

```html
示例 1:

输入: [3,2,3]
输出: 3
示例 2:

输入: [2,2,1,1,1,2,2]
输出: 2
```

对原题的表述存在质疑，众数应该是一组数里面出现比例最高的数，而不是超过n/2
- 暴力
- 哈希
- 排序
- 投票
```java
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length/2];
    }
```

# 22. 实现Trie(前缀树)

[Leetcode #22 (Medium)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)
实现一个 Trie (前缀树)，包含`insert`,`search`, 和 `startsWith` 这三个操作。
说明:

- 你可以假设所有的输入都是由小写字母 a-z 构成的。
- 保证所有输入均为非空字符串。

```html
示例:
Trie trie = new Trie();

trie.insert("apple");
trie.search("apple");   // 返回 true
trie.search("app");     // 返回 false
trie.startsWith("app"); // 返回 true
trie.insert("app");   
trie.search("app");     // 返回 true
```

Trie前缀树 与 哈希表
- [实现Trie(前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/solution/shi-xian-trie-qian-zhui-shu-by-leetcode/)
- 字节跳动一面题
```java
class Trie {
    public boolean isWord;
    public char word;
    public Trie[] tries = new Trie[26];

    /** Initialize your data structure here. */
    public Trie() {
        this.isWord =false;
        this.word=' ';
         
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        char [] array = word.toCharArray();
        Trie node=this;
        for(int i=0;i<array.length;i++){
            if(node.tries[array[i]-'a']==null){
                node.tries[array[i]-'a']=new Trie();     
            }
            node=node.tries[array[i]-'a'];
            node.word=array[i];
            if(i==array.length-1){
                node.isWord=true;
            }    
        }
        
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        char [] array =word.toCharArray();
        Trie node =this;
            for(int i=0;i<array.length;i++){
                if(node.tries[array[i]-'a']!=null){
                    node =node.tries[array[i]-'a'];
                    if(node.word ==array[i])
                        continue;
                    else
                        return false;
                }else
                    return false;
            
            }     
            return node.isWord==true?true:false;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        char [] array =prefix.toCharArray();
        Trie node =this;
            for(int i=0;i<array.length;i++){
                if(node.tries[array[i]-'a']!=null){
                    node =node.tries[array[i]-'a'];
                    if(node.word ==array[i])
                        continue;
                    else
                        return false;
                }
                else
                    return false;

            }
            return true;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
```

# 23. 从前序遍历与中序遍历序列构造二叉树

[Leetcode #105 (Medium)](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
根据一棵树的前序遍历与中序遍历构造二叉树。
注意:
你可以假设树中没有重复的元素。
```html
例如，给出

前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7
```
- 宽度优先搜索BFS
- 深度优先搜索DFS
```java
class Solution {
  // start from first preorder element
  int pre_idx = 0;
  int[] preorder;
  int[] inorder;
  HashMap<Integer, Integer> idx_map = new HashMap<Integer, Integer>();

  public TreeNode helper(int in_left, int in_right) {
    // if there is no elements to construct subtrees
    if (in_left == in_right)
      return null;

    // pick up pre_idx element as a root
    int root_val = preorder[pre_idx];
    TreeNode root = new TreeNode(root_val);

    // root splits inorder list
    // into left and right subtrees
    int index = idx_map.get(root_val);

    // recursion 
    pre_idx++;
    // build left subtree
    root.left = helper(in_left, index);
    // build right subtree
    root.right = helper(index + 1, in_right);
    return root;
  }

  public TreeNode buildTree(int[] preorder, int[] inorder) {
    this.preorder = preorder;
    this.inorder = inorder;

    // build a hashmap value -> its index
    int idx = 0;
    for (Integer val : inorder)
      idx_map.put(val, idx++);
    return helper(0, inorder.length);
  }
}
```


# 24. 每日温度

[Leetcode #739 (Medium)](https://leetcode-cn.com/problems/daily-temperatures/)
根据每日 气温 列表，请重新生成一个列表，对应位置的输入是你需要再等待多久温度才会升高超过该日的天数。如果之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 `temperatures = [73, 74, 75, 71, 69, 72, 76, 73]`，你的输出应该是 `[1, 1, 4, 2, 1, 1, 0, 0]`。

提示：气温 列表长度的范围是`[1, 30000]`。每个气温的值的均为华氏度，都是在 `[30, 100]` 范围内的整数。

[单调栈 逆序遍历](https://leetcode-cn.com/problems/daily-temperatures/solution/javadan-diao-zhan-ni-xu-bian-li-by-hyh-2/)

```java
public int[] dailyTemperatures(int[] T) {
        int[] res = new int[T.length];
        // 单调栈 里面的数 非递增排序 
        Stack<Integer> stack = new Stack();
        // 从后往前遍历
        for(int i = T.length-1; i >= 0; i--){
            // 当前元素比栈顶元素大 出栈 重新调整栈直至满足要求
            while(!stack.isEmpty() && T[i] >= T[stack.peek()]){
                stack.pop();
            }
            // 栈为空 即后面没有比当前天温度高的
            // 不为空 栈顶元素对应的下标减去当前下标即为经过几天后温度比当前天温度高
            res[i] = stack.isEmpty()? 0 :stack.peek()-i;
            // 当前元素进栈
            stack.push(i);
        }
        return res;
}
```

# 25. 数组中的第K个最大元素

[Leetcode #215  (Medium)](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)
在未排序的数组中找到第 **k** 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

```html
示例 1:

输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
示例 2:

输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
```
**说明:**
你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度。
分治法思想 

- 堆
- 快速选择
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        // init heap 'the smallest element first'
        PriorityQueue<Integer> heap =
            new PriorityQueue<Integer>((n1, n2) -> n1 - n2);

        // keep k largest elements in the heap
        for (int n: nums) {
          heap.add(n);
          if (heap.size() > k)
            heap.poll();
        }

        // output
        return heap.poll();        
  }
}
````

# 26. 二叉树的层次遍历

[Leetcode #102 (Medium)](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)
给定一个二叉树，返回其按层次遍历的节点值。 （即逐层地，从左到右访问所有节点）。

```html
例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [9,20],
  [15,7]
]
```

- 层次优先遍历
```java
class Solution {
    List<List<Integer>> levels = new ArrayList<List<Integer>>();

    public void helper(TreeNode node, int level) {
        // start the current level
        if (levels.size() == level)
            levels.add(new ArrayList<Integer>());

         // fulfil the current level
         levels.get(level).add(node.val);

         // process child nodes for the next level
         if (node.left != null)
            helper(node.left, level + 1);
         if (node.right != null)
            helper(node.right, level + 1);
    }
    
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) return levels;
        helper(root, 0);
        return levels;
    }
}
```

# 27. 前K个高频元素

[Leetcode #347 (Medium)](https://leetcode-cn.com/problems/top-k-frequent-elements/))
给定一个非空的整数数组，返回其中出现频率前 k 高的元素。
```html
示例 1:

输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
示例 2:

输入: nums = [1], k = 1
输出: [1]
说明：

- 你可以假设给定的 k 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
- 你的算法的时间复杂度必须优于 O(n log n) , n 是数组的大小。
```

- 堆
```java
class Solution {
  public List<Integer> topKFrequent(int[] nums, int k) {
    // build hash map : character and how often it appears
    HashMap<Integer, Integer> count = new HashMap();
    for (int n: nums) {
      count.put(n, count.getOrDefault(n, 0) + 1);
    }

    // init heap 'the less frequent element first'
    PriorityQueue<Integer> heap =
            new PriorityQueue<Integer>((n1, n2) -> count.get(n1) - count.get(n2));

    // keep k top frequent elements in the heap
    for (int n: count.keySet()) {
      heap.add(n);
      if (heap.size() > k)
        heap.poll();
    }

    // build output list
    List<Integer> top_k = new LinkedList();
    while (!heap.isEmpty())
      top_k.add(heap.poll());
    Collections.reverse(top_k);
    return top_k;
  }
}
```

# 28. 字母异位词分组

[Leetcode #49 (Medium)](https://leetcode-cn.com/problems/group-anagrams/)
给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

```html
输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```
说明：
- 所有输入均为小写字母。
- 不考虑答案输出的顺序。

当且仅当它们的排序字符串相等时，两个字符串是字母异位词。
- [排序数组分类](#https://leetcode-cn.com/problems/group-anagrams/solution/zi-mu-yi-wei-ci-fen-zu-by-leetcode/)
```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs.length == 0) return new ArrayList();
        Map<String, List> ans = new HashMap<String, List>();
        for (String s : strs) {
            char[] ca = s.toCharArray();
            Arrays.sort(ca);
            String key = String.valueOf(ca);
            if (!ans.containsKey(key)) ans.put(key, new ArrayList());
            ans.get(key).add(s);
        }
        return new ArrayList(ans.values());
    }
}
```

#29. 回文子串

[Leetcode #647  (Medium)](https://leetcode-cn.com/problems/palindromic-substrings/)
给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。
具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被计为是不同的子串。
输入的字符串长度不会超过1000。
```html
示例 1:

输入: "abc"
输出: 3
解释: 三个回文子串: "a", "b", "c".
示例 2:

输入: "aaa"
输出: 6
说明: 6个回文子串: "a", "a", "a", "aa", "aa", "aaa".
```
- DP
- 中心拓展法
```java
	 public int countSubstrings(String s) {
    if (s == null || s.length() == 0) {
        return 0;
    }
    int result = 0;
    boolean[][] dp = buildDPForCountSubstrings(s);
    for (int j = 0; j < dp.length; j++) {
        for (int i = 0; i <= j; i++) {
            if (dp[i][j]) {
                result++;
            }
        }
    }
    return result;
}


private boolean[][] buildDPForCountSubstrings(String s) {
    int n = s.length();
    boolean[][] dp = new boolean[n][n];
    //注意i 和j 的边界，只计算上半部分，j - i <= 1是为了处理边界，dp[i + 1][j - 1]是dp[i][j]砍头去尾后的是否是回文
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            if (i == j) {
                dp[i][j] = true;
            } else {
                dp[i][j] = s.charAt(i) == s.charAt(j) && (j - i <= 1 || dp[i + 1][j - 1]);
            }
        }
    }
    return dp;
}
```

#30. 盛最多水的容器

[Leetcode #11 (Medium)](https://leetcode-cn.com/problems/container-with-most-water/)
给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器，且 n 的值至少为 2。
<img src="https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg">

图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。

```html
示例:
输入: [1,8,6,2,5,4,8,3,7]
输出: 49
```

- 暴力法
- 双指针法
```java
public class Solution {
    public int maxArea(int[] height) {
        int maxarea = 0, l = 0, r = height.length - 1;
        while (l < r) {
            maxarea = Math.max(maxarea, Math.min(height[l], height[r]) * (r - l));
            if (height[l] < height[r])
                l++;
            else
                r--;
        }
        return maxarea;
    }
}
```
