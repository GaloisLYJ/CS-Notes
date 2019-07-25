<!-- GFM-TOC -->

* [51. 任务调度器](#51-任务调度器)
* [52. 两数之和](#52-两数之和)
* [53. 最大子序和](#53-最大子序和)
* [54. 爬楼梯](#54-爬楼梯)
* [55. 相交链表](#55-相交链表)
* [56. 二叉树的直径](#56-二叉树的直径)
* [57. 最长连续序列](#57. 最长连续序列)
* [58. 接雨水](#58. 接雨水)
* [59. 字符串解码](#https://leetcode-cn.com/problems/decode-string/)
* [60. 岛屿数量](#https://leetcode-cn.com/problems/number-of-islands/)


  <!-- GFM-TOC -->


# 51. 任务调度器

[Leetcode #621 (Easy)](https://leetcode-cn.com/problems/task-scheduler/)

给定一个用字符数组表示的 CPU 需要执行的任务列表。其中包含使用大写的 A - Z 字母表示的26 种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。CPU 在任何一个单位时间内都可以执行一个任务，或者在待命状态。

然而，两个**相同种类**的任务之间必须有长度为 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。

你需要计算完成所有任务所需要的**最短时间**。

```html
示例 1：

输入: tasks = ["A","A","A","B","B","B"], n = 2
输出: 8
执行顺序: A -> B -> (待命) -> A -> B -> (待命) -> A -> B.
```

**注：**

1. 任务的总个数为 [1, 10000]。
2. n 的取值范围为 [0, 100]。

- 最短路径
``` java
class Solution {
    public int leastInterval(char[] tasks, int n) {
        if(tasks.length==0) return 0;
        int ch[] = new int[256];
        //统计字母出现的频率
        for(char c : tasks){
            ch[c]++;
        }
        //获取出现最多的次数
        int max = Integer.MIN_VALUE;
        for(int i :ch){
            max =Math.max(max,i);
        }
        //得到一共有几个字母都以最高频次出现
        int  count=0;
        for(int i :ch){
            if(i==max) ++count;
        }
        return Math.max((n+1)*(max-1)+count,tasks.length);
    }
}
```


# 52. 两数之和

[Leetcode #1 (Easy)](https://leetcode-cn.com/problems/two-sum/)
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

```html
示例:

给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```

- 两遍哈希表
```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], i);
        }
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement) && map.get(complement) != i) {
                return new int[] { i, map.get(complement) };
            }
        }
        throw new IllegalArgumentException("No two sum solution");
    }
}
```

#53. 最大子序和

[Leetcode #53 (Easy)](https://leetcode-cn.com/problems/maximum-subarray/)
给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

```html
示例:

输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

- [动态规划](https://leetcode-cn.com/problems/maximum-subarray/solution/hua-jie-suan-fa-53-zui-da-zi-xu-he-by-guanpengchn/)
- 分治法
```java
class Solution {
    public int maxSubArray(int[] nums) {
        int ans = nums[0];
        int sum = 0;
        for(int num: nums) {
            if(sum > 0) {
                sum += num;
            } else {
                sum = num;
            }
            ans = Math.max(ans, sum);
        }
        return ans;
    }
}
```

#54. 爬楼梯

[Leetcode #70 (Easy)](https://leetcode-cn.com/problems/climbing-stairs/)

假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**注意：**给定 *n* 是一个正整数。

```html
示例 1：

输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶
示例 2：

输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶
```
- 暴力法
- 记忆化递归
- [动态规划](https://leetcode-cn.com/problems/climbing-stairs/solution/pa-lou-ti-by-leetcode/)

```java
public class Solution {
    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```

# 55. 相交链表

[Leetcode #160 (Easy)](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

编写一个程序，找到两个单链表相交的起始节点。

如下面的两个链表：

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png">

在节点 c1 开始相交。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_1.png"> 

示例 1：

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_1.png">

输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。


示例 2：

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_2.png">

输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Reference of the node with value = 2
输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。


示例 3：

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_3.png">

输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
输入解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
解释：这两个链表不相交，因此返回 null。


注意：

- 如果两个链表没有交点，返回 null.
- 在返回结果后，两个链表仍须保持原有的结构。
- 可假定整个链表结构中没有循环。
- 程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。

- 暴力法

- 哈希表法

- 双指针法

- [图解法](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/solution/tu-jie-xiang-jiao-lian-biao-by-user7208t/)

  ```java
  public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
      if (headA == null || headB == null) return null;
      ListNode pA = headA, pB = headB;
      while (pA != pB) {
          pA = pA == null ? headB : pA.next;
          pB = pB == null ? headA : pB.next;
      }
      return pA;
  }
  ```

  

# 56.  二叉树的直径

[Leetcode #543 (Easy)](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过根结点。

**示例 :**
给定二叉树

```html
          1
         / \
        2   3
       / \     
      4   5    
```

返回 **3**, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。

- [递归](https://leetcode-cn.com/problems/diameter-of-binary-tree/solution/javade-di-gui-jie-fa-by-lyl0724-2/)

> ```
> 一个节点的最大直径 = 它左树的高度 +  它右树的高度
> ```

```java
class Solution {
    //设置一个类变量，用于记录最大直径
    private int max = 0;
    
    public int diameterOfBinaryTree(TreeNode root) {
        depth(root);
        return max;
    }
    
    private int depth(TreeNode root){
        if(root == null){
            return 0;
        }
        int leftDepth = depth(root.left);
        int rightDepth = depth(root.right);
        //max记录当前的最大直径
        max = Math.max(leftDepth + rightDepth, max);
        //由于我计算的直径是左树高度+右树高度，所以这里返回当前树的高度，以供使用
        return Math.max(leftDepth, rightDepth) + 1;
    }
}
```

# 57. 最长连续序列

[Leetcode #128 (Difficult)](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

给定一个未排序的整数数组，找出最长连续序列的长度。

要求算法的时间复杂度为 *O(n)*。

```html
输入: [100, 4, 200, 1, 3, 2]
输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

- 暴力法
- 排序
- 哈希表和连续空间构造

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> num_set = new HashSet<Integer>();
        for (int num : nums) {
            num_set.add(num);
        }

        int longestStreak = 0;

        for (int num : num_set) {
            if (!num_set.contains(num-1)) {
                int currentNum = num;
                int currentStreak = 1;

                while (num_set.contains(currentNum+1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }

                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }

        return longestStreak;
    }
}
```



# 58. 接雨水

[Leetcode #42 (Difficult)](https://leetcode-cn.com/problems/trapping-rain-water/)

给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png">

上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。

```html
示例:

输入: [0,1,0,2,1,0,1,3,2,1,2,1]
输出: 6
```

- 暴力法
- 动态编程
- 栈的应用
- 双指针

```java
    public int trap(int[] height)
    {
        int ans = 0;
        int size = height.length;
        for (int i = 1; i < size - 1; i++) {
            int max_left = 0, max_right = 0;
            for (int j = i; j >= 0; j--) { //Search the left part for max bar size
                max_left = Math.max(max_left, height[j]);
            }
            for (int j = i; j < size; j++) { //Search the right part for max bar size
                max_right = Math.max(max_right, height[j]);
            }
            ans += Math.min(max_left, max_right) - height[i];
        }
        return ans;
    }
```

# 59. 字符串解码

[Leetcode #394 (Medium)](#https://leetcode-cn.com/problems/decode-string/)

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: `k[encoded_string]`，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 `3a` 或 `2[4]` 的输入。

```html
s = "3[a]2[bc]", 返回 "aaabcbc".
s = "3[a2[c]]", 返回 "accaccacc".
s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".
```

- 括号匹配与栈

  ```java
  class Solution {
      public String decodeString(String s) {
          StringBuilder res = new StringBuilder();
          Stack<Integer> numStack = new Stack<>();
          Stack<String> strStack = new Stack<>();
          String tempStr = null;
          for (int i = 0; i < s.length(); i++) {
              char c = s.charAt(i);
              if (s.charAt(i) == ']') {
                  String str = strStack.pop();
                  int num = numStack.pop();
                  String nowStr = repeatStr(str, num);
                  if (!numStack.isEmpty()) {
                     StringBuilder  builder = new StringBuilder();
                     builder.append(strStack.peek());
                     builder.append(nowStr);
                     int m = i + 1;
                     while (s.charAt(m) != ']' && !('0' < s.charAt(m) && '9' >= s.charAt(m))) {
                         m++;
                     }
                     builder.append(s.substring(i + 1, m));
                     strStack.set(strStack.size() - 1, builder.toString());
                     i = m - 1;
                  } else {
                      tempStr = null;
                      res.append(nowStr);
                  }
              } else if ('0' <= c && '9' >= c) {
                  int m = i + 1;
                  while ('0' <= s.charAt(m) && '9' >= s.charAt(m)) {
                      m++;
                  }
                  numStack.push(Integer.parseInt(s.substring(i, m)));
                  i = m - 1;
                  int k =  i + 2;
                  while (s.charAt(k) != ']' && !('0' <= s.charAt(k) && '9' >= s.charAt(k)))  {
                      k++;
                  }
                  strStack.push(s.substring(i+2, k));
                  i = k - 1;
              } else if (numStack.isEmpty()) {
                  res.append(s.charAt(i));
              }
          }
          return res.toString();
      }
  
      private String repeatStr(String str, int num) {
          StringBuilder sb = new StringBuilder();
          if (num <= 0) {
              return "";
          }
          for (int i = 0; i < num; i++) {
              sb.append(str);
          }
          return sb.toString();
      }
  }
  ```

  

# 60. 岛屿数量

[Leetcode #200 (Medium)](https://leetcode-cn.com/problems/number-of-islands/)

给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

```html
示例 1:

输入:
11110
11010
11000
00000

输出: 1
示例 2:

输入:
11000
11000
00100
00011

输出: 3
```

- [深度优先搜索](#https://leetcode-cn.com/problems/number-of-islands/solution/dao-yu-shu-liang-by-leetcode/)

- 广度优先搜索

- 并查集

  ```java
  class Solution {
    void dfs(char[][] grid, int r, int c) {
      int nr = grid.length;
      int nc = grid[0].length;
  
      if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] == '0') {
        return;
      }
  
      grid[r][c] = '0';
      dfs(grid, r - 1, c);
      dfs(grid, r + 1, c);
      dfs(grid, r, c - 1);
      dfs(grid, r, c + 1);
    }
  
    public int numIslands(char[][] grid) {
      if (grid == null || grid.length == 0) {
        return 0;
      }
  
      int nr = grid.length;
      int nc = grid[0].length;
      int num_islands = 0;
      for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < nc; ++c) {
          if (grid[r][c] == '1') {
            ++num_islands;
            dfs(grid, r, c);
          }
        }
      }
  
      return num_islands;
    }
  }
  ```

  