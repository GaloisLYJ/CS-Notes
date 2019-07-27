<!-- GFM-TOC -->

* [71. 目标和](#71-目标和)
* [72. 二叉树的序列化与反序列化](#72-二叉树的序列化与反序列化)
* [63. 最长上升子序列](#63-最长上升子序列)
* [64. LRU缓存机制](#64. LRU缓存机制)
* [65. 滑动窗口最大值](#65. 滑动窗口最大值)
* [66. 单词拆分](#66. 单词拆分)
* [67. 环形链表II](#67. 环形链表II)
* [68. 环形链表](#68. 环形链表)
* [69. 打家劫舍](#69. 打家劫舍)
* [70. 删除无效的括号](#70. 删除无效的括号)


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

# 73. 最长上升子序列

[Leetcode #300 (Medium)](<https://leetcode-cn.com/problems/longest-increasing-subsequence/>)

给定一个无序的整数数组，找到其中最长上升子序列的长度。

```html
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
```

说明:

- 可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
- 你算法的时间复杂度应该为 O(n2) 。

进阶: 你能将算法的时间复杂度降低到 O(n log n) 吗?

- 暴力法
- 带记忆的递归
- 动态规划
- 动态规划和二分搜索 

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        int len = 0;
        for (int num : nums) {
            int i = Arrays.binarySearch(dp, 0, len, num);
            if (i < 0) {
                i = -(i + 1);
            }
            dp[i] = num;
            if (i == len) {
                len++;
            }
        }
        return len;
    }
}
```

# 64. LRU缓存机制

[Leetcode #146 (Medium)](<https://leetcode-cn.com/problems/lru-cache/>)

运用你所掌握的数据结构，设计和实现一个  `LRU (最近最少使用) 缓存机制`。它应该支持以下操作： 获取数据 `get` 和 写入数据 `put` 。

获取数据 `get(key)` - 如果密钥 `(key)` 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
写入数据 `put(key, value)` - 如果密钥不存在，则写入其数据值。当缓存容量达到上限时，它应该在写入新数据之前删除最近最少使用的数据值，从而为新的数据值留出空间。

**进阶:**

你是否可以在 **O(1)** 时间复杂度内完成这两种操作？

```html
LRUCache cache = new LRUCache( 2 /* 缓存容量 */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // 返回  1
cache.put(3, 3);    // 该操作会使得密钥 2 作废
cache.get(2);       // 返回 -1 (未找到)
cache.put(4, 4);    // 该操作会使得密钥 1 作废
cache.get(1);       // 返回 -1 (未找到)
cache.get(3);       // 返回  3
cache.get(4);       // 返回  4
```

- 有序字典
- 哈希表+双向链表

```java
class LRUCache extends LinkedHashMap<Integer, Integer>{
    private int capacity;
    
    public LRUCache(int capacity) {
        super(capacity, 0.75F, true);
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.getOrDefault(key, -1);
    }

    public void put(int key, int value) {
        super.put(key, value);
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity; 
    }
}
```

# 65. 滑动窗口最大值

[Leetcode #65 (Difficult)](<https://leetcode-cn.com/problems/sliding-window-maximum/>)

给定一个数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口 k 内的数字。滑动窗口每次只向右移动一位。

返回滑动窗口最大值。

```html
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**注意：**

你可以假设 *k* 总是有效的，1 ≤ k ≤ 输入数组的大小，且输入数组不为空。

**进阶：**

你能在线性时间复杂度内解决此题吗？

- 暴力法
- 双向队列
- 动态规划

```java
class Solution {
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

# 66. 单词拆分

[Leetcode #139 (Medium)](<https://leetcode-cn.com/problems/word-break/>)

给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

说明：

- 拆分时可以重复使用字典中的单词。
- 你可以假设字典中没有重复的单词。

```html
示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
示例 2：

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
示例 3：

输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

- 暴力
- 记忆化回溯
- 宽度优先搜索
- 动态规划

```java
public class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordDictSet=new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
}
```

# 67. 环形链表II

[Leetcode #142 (Medium)](<https://leetcode-cn.com/problems/linked-list-cycle-ii/>)

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。

说明：不允许修改给定的链表。

```html
示例 1：

输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1
解释：链表中有一个环，其尾部连接到第二个节点。
```

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png">

```html
示例 2：

输入：head = [1,2], pos = 0
输出：tail connects to node index 0
解释：链表中有一个环，其尾部连接到第一个节点。
```

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test2.png">

```html
示例 3：

输入：head = [1], pos = -1
输出：no cycle
解释：链表中没有环。
```

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test3.png">

- 哈希表
- [Floyd算法](<https://leetcode-cn.com/problems/linked-list-cycle-ii/solution/huan-xing-lian-biao-ii-by-leetcode/>)

```java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    private ListNode getIntersect(ListNode head) {
        ListNode tortoise = head;
        ListNode hare = head;

        // A fast pointer will either loop around a cycle and meet the slow
        // pointer or reach the `null` at the end of a non-cyclic list.
        while (hare != null && hare.next != null) {
            tortoise = tortoise.next;
            hare = hare.next.next;
            if (tortoise == hare) {
                return tortoise;
            }
        }

        return null;
}

    public ListNode detectCycle(ListNode head) {
        if (head == null) {
            return null;
        }

        // If there is a cycle, the fast/slow pointers will intersect at some
        // node. Otherwise, there is no cycle, so we cannot find an e***ance to
        // a cycle.
        ListNode intersect = getIntersect(head);
        if (intersect == null) {
            return null;
        }

        // To find the e***ance to the cycle, we have two pointers traverse at
        // the same speed -- one from the front of the list, and the other from
        // the point of intersection.
        ListNode ptr1 = head;
        ListNode ptr2 = intersect;
        while (ptr1 != ptr2) {
            ptr1 = ptr1.next;
            ptr2 = ptr2.next;
        }

        return ptr1;
    }
}
```

# 68. 环形链表

[Leetcode #141 (Easy)](<https://leetcode-cn.com/problems/linked-list-cycle/>)

给定一个链表，判断链表中是否有环。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。

实例说明与上题同。

- 哈希表
- 双指针

```java
public boolean hasCycle(ListNode head) {
    if (head == null || head.next == null) {
        return false;
    }
    ListNode slow = head;
    ListNode fast = head.next;
    while (slow != fast) {
        if (fast == null || fast.next == null) {
            return false;
        }
        slow = slow.next;
        fast = fast.next.next;
    }
    return true;
}
```

# 69. 打家劫舍

[Leetcode #198 (Easy)](<https://leetcode-cn.com/problems/house-robber/>)

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。**

给定一个代表每个房屋存放金额的非负整数数组，计算你**在不触动警报装置的情况下**，能够偷窃到的最高金额。

```html
示例 1:

输入: [1,2,3,1]
输出: 4
解释: 偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
示例 2:

输入: [2,7,9,3,1]
输出: 12
解释: 偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

- [动态规划真有这么好？](<https://leetcode-cn.com/problems/house-robber/solution/da-jia-jie-she-by-leetcode/>)

```java
public int rob(int[] num) {
    int prevMax = 0;
    int currMax = 0;
    for (int x : num) {
        int temp = currMax;
        currMax = Math.max(prevMax + x, currMax);
        prevMax = temp;
    }
    return currMax;
}
```

# 70. 删除无效的括号

[Leetcode #301 (Difficult)](<https://leetcode-cn.com/problems/remove-invalid-parentheses/>)

删除最小数量的无效括号，使得输入的字符串有效，返回所有可能的结果。

**说明:** 输入可能包含了除 `(` 和 `)` 以外的字符。

```html
示例 1:

输入: "()())()"
输出: ["()()()", "(())()"]
示例 2:

输入: "(a)())()"
输出: ["(a)()()", "(a())()"]
示例 3:

输入: ")("
输出: [""]
```

- 回溯
- 有限的回溯

```java
class Solution {

  private Set<String> validExpressions = new HashSet<String>();

  private void recurse(
      String s,
      int index,
      int leftCount,
      int rightCount,
      int leftRem,
      int rightRem,
      StringBuilder expression) {

    // If we reached the end of the string, just check if the resulting expression is
    // valid or not and also if we have removed the total number of left and right
    // parentheses that we should have removed.
    if (index == s.length()) {
      if (leftRem == 0 && rightRem == 0) {
        this.validExpressions.add(expression.toString());
      }

    } else {
      char character = s.charAt(index);
      int length = expression.length();

      // The discard case. Note that here we have our pruning condition.
      // We don't recurse if the remaining count for that parenthesis is == 0.
      if ((character == '(' && leftRem > 0) || (character == ')' && rightRem > 0)) {
        this.recurse(
            s,
            index + 1,
            leftCount,
            rightCount,
            leftRem - (character == '(' ? 1 : 0),
            rightRem - (character == ')' ? 1 : 0),
            expression);
      }

      expression.append(character);

      // Simply recurse one step further if the current character is not a parenthesis.
      if (character != '(' && character != ')') {

        this.recurse(s, index + 1, leftCount, rightCount, leftRem, rightRem, expression);

      } else if (character == '(') {

        // Consider an opening bracket.
        this.recurse(s, index + 1, leftCount + 1, rightCount, leftRem, rightRem, expression);

      } else if (rightCount < leftCount) {

        // Consider a closing bracket.
        this.recurse(s, index + 1, leftCount, rightCount + 1, leftRem, rightRem, expression);
      }

      // Delete for backtracking.
      expression.deleteCharAt(length);
    }
  }

  public List<String> removeInvalidParentheses(String s) {

    int left = 0, right = 0;

    // First, we find out the number of misplaced left and right parentheses.
    for (int i = 0; i < s.length(); i++) {

      // Simply record the left one.
      if (s.charAt(i) == '(') {
        left++;
      } else if (s.charAt(i) == ')') {
        // If we don't have a matching left, then this is a misplaced right, record it.
        right = left == 0 ? right + 1 : right;

        // Decrement count of left parentheses because we have found a right
        // which CAN be a matching one for a left.
        left = left > 0 ? left - 1 : left;
      }
    }

    this.recurse(s, 0, 0, 0, left, right, new StringBuilder());
    return new ArrayList<String>(this.validExpressions);
  }
}
```

