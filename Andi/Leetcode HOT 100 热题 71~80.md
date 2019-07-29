<!-- GFM-TOC -->

* [71. 目标和](#71-目标和)

* [72. 二叉树的序列化与反序列化](#72-二叉树的序列化与反序列化)

* [73. 有效的括号](#73. 有效的括号)

* [74. 单词搜索](#74. 单词搜索)

* [75. 找到字符串中所有字母异位词](#75. 找到字符串中所有字母异位词)

* [76. 回文链表](#76. 回文链表)

* [77. 柱状图中最大矩形](#77. 柱状图中最大的矩形)

* [78. 最大正方形](#78. 最大正方形)

* [79. 搜索二维矩阵II](#79. 搜索二维矩阵II)

* [80. 在排序数组中查找元素的第一个和最后一个位置](#80. 在排序数组中查找元素的第一个和最后一个位置)

* 


  <!-- GFM-TOC -->


# 71. 目标和

[Leetcode #494 (Medium)](<https://leetcode-cn.com/problems/target-sum/>)

给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。

返回可以使最终数组和为目标数 S 的所有添加符号的方法数。

```html
示例 1:

输入: nums: [1, 1, 1, 1, 1], S: 3
输出: 5
解释: 

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

一共有5种方法让最终目标和为3。
```

- 01背包 动态规划
- C++

```c++
int findTargetSumWays(vector<int>& nums, int S) {
            int sum=accumulate(nums.begin(),nums.end(),0);
            if(sum<S||sum+S&1) return 0;
            //special case:
            int nZeros=0;
            for(auto it=nums.begin();it!=nums.end();){
                if(*it==0){
                    nZeros++;
                    it=nums.erase(it);
                }else ++it;
            }

            //开始动规求解
            int row=nums.size();
            int x=sum+S>>1;
            vector<vector<int>>dp(row+1,vector<int>(x+1,0));
            dp[0][0]=1;
            for(int i=0;i<row;i++){
                dp[i+1][0]=1;//注意第一列初始为1，表示容量为0时，有一种方式，即每个数值都不选
                for(int j=0;j<x;j++){
                    if(nums[i]<=j+1) dp[i+1][j+1]=dp[i][j+1]+dp[i][j+1-nums[i]];//选不选这个数
                    else dp[i+1][j+1]=dp[i][j+1];//放不下这个数，只能不选
                }
            }
            return dp[row][x]*(1<<nZeros);
        }
```


# 72. 二叉树的序列化与反序列化

[Leetcode #297 (Difficult)](<https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/>)
序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

```html
你可以将以下二叉树：

    1
   / \
  2   3
     / \
    4   5

序列化为 "[1,2,3,null,null,4,5]"
```

提示: 这与 LeetCode 目前使用的方式一致，详情请参阅 [LeetCode 序列化二叉树的格式](<https://support.leetcode-cn.com/hc/kb/category/1018267/>)。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。

说明: 不要使用类的成员 / 全局 / 静态变量来存储状态，你的序列化和反序列化算法应该是无状态的。

- 深度优先搜索
```java
import java.util.regex.Matcher;
import java.util.regex.Pattern;
public class Codec {
    //按层遍历二叉树，用一个队列保存每一层的节点
    public String serialize(TreeNode root) {
        if (root == null)
            return "";
        StringBuilder builder = new StringBuilder();
        Deque<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int count = queue.size();
            while (count-- > 0) {
                TreeNode node = queue.removeFirst();
                //这个优化点很重要，当遍历到null时，不要往队列写入null子节点，减少生成的字符串大小
                if (node != null) {
                    queue.add(node.left);
                    queue.add(node.right);
                    builder.append(",").append(node.val);
                } else {
                    builder.append(",null");
                }
            }
        }
        //删除第一个逗号
        builder.deleteCharAt(0);
        //下面的代码通过正则匹配字符串末尾的null,这些null可以省略
        //Pattern pattern = Pattern.compile("[,null]+$");
        //Matcher matcher = pattern.matcher(builder.toString());
        //return matcher.replaceFirst("");
        return builder.toString();
    }
    //反序列化的时候，用两个指针，第一个指向父节点，第二个指向子节点
    //由于字符串是按层遍历，同一层的节点相邻，子节点排在父节点后面
    //用一个队列记录按层遍历的树节点，不包含null,队列头节点为整个树的根节点
    public TreeNode deserialize(String data) {
        if (data == null || data.length() == 0)
            return null;
        String[] values = data.split(",");
        List<TreeNode> list = new LinkedList<>();
        TreeNode head = createNode(values[0]);
        list.add(head);
        int rootIndex = 0;
        int valueIndex = 1;
        while (rootIndex < list.size()) {
            TreeNode root = list.get(rootIndex++);
            if (valueIndex < values.length){
                root.left = createNode(values[valueIndex++]);
                root.right = createNode(values[valueIndex++]);
            }                
            if (root.left != null)
                list.add(root.left);
            if (root.right != null)
                list.add(root.right);
        }
        return head;
    }

    private TreeNode createNode(String str) {
        if (str == null) {
            return null;
        }
        if (str.equalsIgnoreCase("null")) {
            return null;
        } else {
            int integer = Integer.parseInt(str);
            return new TreeNode(integer);
        }
    }
}
```

# 73. 有效的括号

[Leetcode #20 (Easy)](https://leetcode-cn.com/problems/valid-parentheses/)

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。

```html
示例 1:

输入: "()"
输出: true
示例 2:

输入: "()[]{}"
输出: true
示例 3:

输入: "(]"
输出: false
示例 4:

输入: "([)]"
输出: false
示例 5:

输入: "{[]}"
输出: true
```

- 栈

```java
class Solution {

  // Hash table that takes care of the mappings.
  private HashMap<Character, Character> mappings;

  // Initialize hash map with mappings. This simply makes the code easier to read.
  public Solution() {
    this.mappings = new HashMap<Character, Character>();
    this.mappings.put(')', '(');
    this.mappings.put('}', '{');
    this.mappings.put(']', '[');
  }

  public boolean isValid(String s) {

    // Initialize a stack to be used in the algorithm.
    Stack<Character> stack = new Stack<Character>();

    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);

      // If the current character is a closing bracket.
      if (this.mappings.containsKey(c)) {

        // Get the top element of the stack. If the stack is empty, set a dummy value of '#'
        char topElement = stack.empty() ? '#' : stack.pop();

        // If the mapping for this bracket doesn't match the stack's top element, return false.
        if (topElement != this.mappings.get(c)) {
          return false;
        }
      } else {
        // If it was an opening bracket, push to the stack.
        stack.push(c);
      }
    }

    // If the stack still contains elements, then it is an invalid expression.
    return stack.isEmpty();
  }
}
```

# 74. 单词搜索

[Leetcode #79 (Medium)](https://leetcode-cn.com/problems/word-search/)

给定一个二维网格和一个单词，找出该单词是否存在于网格中。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

```html
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

给定 word = "ABCCED", 返回 true.
给定 word = "SEE", 返回 true.
给定 word = "ABCB", 返回 false.
```

- [二维平面上的回溯法](https://leetcode-cn.com/problems/two-sum/solution/zai-er-wei-ping-mian-shang-shi-yong-hui-su-fa-pyth/)

```java
public class Solution {

    private boolean[][] marked;

    //        x-1,y
    // x,y-1  x,y    x,y+1
    //        x+1,y
    private int[][] direction = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    // 盘面上有多少行
    private int m;
    // 盘面上有多少列
    private int n;
    private String word;
    private char[][] board;

    public boolean exist(char[][] board, String word) {
        m = board.length;
        if (m == 0) {
            return false;
        }
        n = board[0].length;
        marked = new boolean[m][n];
        this.word = word;
        this.board = board;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (dfs(i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(int i, int j, int start) {
        if (start == word.length() - 1) {
            return board[i][j] == word.charAt(start);
        }
        if (board[i][j] == word.charAt(start)) {
            marked[i][j] = true;
            for (int k = 0; k < 4; k++) {
                int newX = i + direction[k][0];
                int newY = j + direction[k][1];
                if (inArea(newX, newY) && !marked[newX][newY]) {
                    if (dfs(newX, newY, start + 1)) {
                        return true;
                    }
                }
            }
            marked[i][j] = false;
        }
        return false;
    }

    private boolean inArea(int x, int y) {
        return x >= 0 && x < m && y >= 0 && y < n;
    }

    public static void main(String[] args) {

//        char[][] board =
//                {
//                        {'A', 'B', 'C', 'E'},
//                        {'S', 'F', 'C', 'S'},
//                        {'A', 'D', 'E', 'E'}
//                };
//
//        String word = "ABCCED";


        char[][] board = {{'a', 'b'}};
        String word = "ba";
        Solution solution = new Solution();
        boolean exist = solution.exist(board, word);
        System.out.println(exist);
    }
}
```

# 75. 找到字符串中所有字母异位词

[Leetcode #438 (Easy)](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。

字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。

说明：

字母异位词指字母相同，但排列不同的字符串。
不考虑答案输出的顺序。

```html
示例 1:

输入:
s: "cbaebabacd" p: "abc"

输出:
[0, 6]

解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的字母异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的字母异位词。
 示例 2:

输入:
s: "abab" p: "ab"

输出:
[0, 1, 2]

解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的字母异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的字母异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的字母异位词。
```

- [双指针 滑动窗口](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/solution/shuang-zhi-zhen-hua-kuai-by-mr_tao/)

```java
public List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new ArrayList<>();
        int[] p_letter = new int[26];
        for (int i = 0; i < p.length(); i++) {//记录p里面的数字分别有几个
            p_letter[p.charAt(i) - 'a']++;
        }
        int start = 0;
        int end = 0;
        int[] between_letter = new int[26];//记录两个指针之间的数字都有几个
        while (end < s.length()) {
            int c = s.charAt(end++) - 'a';//每一次拿到end指针对应的字母
            between_letter[c]++;//让这个字母的数量+1

            //如果这个字母的数量比p里面多了,说明这个start坐标需要排除
            while (between_letter[c] > p_letter[c]) {
                between_letter[s.charAt(start++) - 'a']--;
            }
            if (end - start == p.length()) {
                result.add(start);
            }
        }
        return result;
    }  class Solution {
  public int[] maxSlidingWindow(int[] nums, int k) {
    int n = nums.length;
    if (n * k == 0) return new int[0];
    if (k == 1) return nums;

    int [] left = new int[n];
    left[0] = nums[0];
    int [] right = new int[n];
    right[n - 1] = nums[n - 1];
    for (int i = 1; i < n; i++) {
      // from left to right
      if (i % k == 0) left[i] = nums[i];  // block_start
      else left[i] = Math.max(left[i - 1], nums[i]);

      // from right to left
      int j = n - i - 1;
      if ((j + 1) % k == 0) right[j] = nums[j];  // block_end
      else right[j] = Math.max(right[j + 1], nums[j]);
    }

    int [] output = new int[n - k + 1];
    for (int i = 0; i < n - k + 1; i++)
      output[i] = Math.max(left[i + k - 1], right[i]);

    return output;
  }
}
```

# 76. 回文链表

[Leetcode #234 (Easy)](https://leetcode-cn.com/problems/palindrome-linked-list/)

请判断一个链表是否为回文链表。

```html
示例 1:

输入: 1->2
输出: false
示例 2:

输入: 1->2->2->1
输出: true

```

进阶：
你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？

- 快慢指针

```java
public boolean isPalindrome(ListNode head) {
  if(head == null || head.next == null) return true;
  ListNode slow = head, fast = head.next, pre = null, prepre = null;
  while(fast != null && fast.next != null) {
    //反转前半段链表
    pre = slow;
    slow = slow.next;
    fast = fast.next.next;
    //先移动指针再来反转
    pre.next = prepre;
    prepre = pre;
  }
  ListNode p2 = slow.next;
  slow.next = pre;
  ListNode p1 = fast == null? slow.next : slow;
  while(p1 != null) {
    if(p1.val != p2.val)
      return false;
    p1 = p1.next;
    p2 = p2.next;
  }
  return true;
}
```

# 77. 柱状图中最大的矩形

[Leetcode #77 (Difficult)](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram.png">

以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 `[2,1,5,6,2,3]`。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram_area.png">

图中阴影部分为所能勾勒出的最大矩形面积，其面积为 `10` 个单位。

```html
示例:

输入: [2,1,5,6,2,3]
输出: 10
```

- 优化的分治

```java
        public int largestRectangleArea(int[] heights) {
            Stack < Integer > stack = new Stack < > ();
            stack.push(-1);
            int maxarea = 0;
            for (int i = 0; i < heights.length; ++i) {
                while (stack.peek() != -1 && heights[stack.peek()] >= heights[i])
                    maxarea = Math.max(maxarea, heights[stack.pop()] * (i - stack.peek() - 1));
                stack.push(i);
            }
            while (stack.peek() != -1)
                maxarea = Math.max(maxarea, heights[stack.pop()] * (heights.length - stack.peek() -1));
            return maxarea;
        }
```

# 78. 最大正方形

[Leetcode #221 (Medium)](https://leetcode-cn.com/problems/maximal-square/)

在一个由 0 和 1 组成的二维矩阵内，找到只包含 1 的最大正方形，并返回其面积。

```java
示例:

输入: 

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

输出: 4
```

- 动态规划

- 动态规划优化

  ```java
      public int maximalSquare(char[][] matrix) {
          int rows = matrix.length, cols = rows > 0 ? matrix[0].length : 0;
          int[][] dp = new int[rows + 1][cols + 1];
          int maxsqlen = 0;
          for (int i = 1; i <= rows; i++) {
              for (int j = 1; j <= cols; j++) {
                  if (matrix[i-1][j-1] == '1'){
                      dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
                      maxsqlen = Math.max(maxsqlen, dp[i][j]);
                  }
              }
          }
          return maxsqlen * maxsqlen;
      }
  ```

# 79. 打家劫舍

[Leetcode #56 (Medium)](https://leetcode-cn.com/problems/merge-intervals/)

给出一个区间的集合，请合并所有重叠的区间。

```html
示例 1:

输入: [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
示例 2:

输入: [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
```

- [连通块](https://leetcode-cn.com/problems/merge-intervals/solution/java-shi-jian-fu-za-du-onlogn-by-horanol/)
- 排序

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length == 0 || intervals[0].length == 0)
            return new int[0][0];
        LinkedList<Interval> list = new LinkedList<>();
        for (int i = 0; i < intervals.length; i++) {
            int[] nums = intervals[i];
            Interval in = new Interval(nums[0], nums[1]);
            list.add(in);
        }
        //按照区间第一个数字排序
        list.sort(Comparator.comparingInt(o -> o.num1));
        List<Interval> resList = new ArrayList<>();
        Interval first = list.removeFirst();
        while (!list.isEmpty()) {
            Interval second = list.removeFirst();
            if (first.num2 < second.num1) {
                //两个区间不相交
                resList.add(first);
                first = second;
            } else {
                //合并两个区间
                first = new Interval(first.num1, Math.max(first.num2, second.num2));
            }
        }
        resList.add(first);
        int[][] res = new int[resList.size()][2];
        for (int i = 0; i < resList.size(); i++) {
            res[i][0] = resList.get(i).num1;
            res[i][1] = resList.get(i).num2;
        }
        return res;
    }

    class Interval {
        int num1;
        int num2;

        Interval(int num1, int num2) {
            this.num1 = num1;
            this.num2 = num2;
        }
    }
}
```

# 79. 搜索二维矩阵

[Leetcode #240 (Medium)](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

```html
示例:

现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```

给定 target = `5`，返回 `true`。

给定 target = `20`，返回 `false`。

- 二分法搜索
- 搜索空间的缩减

```java
class Solution {
    private boolean binarySearch(int[][] matrix, int target, int start, boolean vertical) {
        int lo = start;
        int hi = vertical ? matrix[0].length-1 : matrix.length-1;

        while (hi >= lo) {
            int mid = (lo + hi)/2;
            if (vertical) { // searching a column
                if (matrix[start][mid] < target) {
                    lo = mid + 1;
                } else if (matrix[start][mid] > target) {
                    hi = mid - 1;
                } else {
                    return true;
                }
            } else { // searching a row
                if (matrix[mid][start] < target) {
                    lo = mid + 1;
                } else if (matrix[mid][start] > target) {
                    hi = mid - 1;
                } else {
                    return true;
                }
            }
        }

        return false;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        // an empty matrix obviously does not contain `target`
        if (matrix == null || matrix.length == 0) {
            return false;
        }

        // iterate over matrix diagonals
        int shorterDim = Math.min(matrix.length, matrix[0].length);
        for (int i = 0; i < shorterDim; i++) {
            boolean verticalFound = binarySearch(matrix, target, i, true);
            boolean horizontalFound = binarySearch(matrix, target, i, false);
            if (verticalFound || horizontalFound) {
                return true;
            }
        }
        
        return false; 
    }
}
```

#80. 在排序数组中查找元素的第一个和最后一个位置

[Leetcode #34. (Medium)](#https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

给定一个按照升序排列的整数数组 `nums`，和一个目标值 `target`。找出给定目标值在数组中的开始位置和结束位置。

你的算法时间复杂度必须是 O(log n) 级别。

如果数组中不存在目标值，返回 `[-1, -1]`。

```html
示例 1:

输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]
示例 2:

输入: nums = [5,7,7,8,8,10], target = 6
输出: [-1,-1]
```

- 线性扫描
- 二分查找

```java
class Solution {
    // returns leftmost (or rightmost) index at which `target` should be
    // inserted in sorted array `nums` via binary search.
    private int extremeInsertionIndex(int[] nums, int target, boolean left) {
        int lo = 0;
        int hi = nums.length;

        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (nums[mid] > target || (left && target == nums[mid])) {
                hi = mid;
            }
            else {
                lo = mid+1;
            }
        }

        return lo;
    }

    public int[] searchRange(int[] nums, int target) {
        int[] targetRange = {-1, -1};

        int leftIdx = extremeInsertionIndex(nums, target, true);

        // assert that `leftIdx` is within the array bounds and that `target`
        // is actually in `nums`.
        if (leftIdx == nums.length || nums[leftIdx] != target) {
            return targetRange;
        }

        targetRange[0] = leftIdx;
        targetRange[1] = extremeInsertionIndex(nums, target, false)-1;

        return targetRange;
    }
}
```

