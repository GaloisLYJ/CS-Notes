<!-- GFM-TOC -->
* [41. 找到所有数组中消失的数字](#41-找到所有数组中消失的数字)
* [42. 完全平方数](#42-完全平方数)
* [43. 买卖股票的最佳时机](#43-买卖股票的最佳时机)
* [44. 电话号码的字母组合](#44-电话号码的字母组合)
* [45. 最小栈](#45-最小栈)
* [46. 除法求值](#46-除法求值)
* [47. 最佳买卖股票时机含冷冻期](#47-最佳买卖股票时机含冷冻期)
* [48. 对称二叉树](#48-对称二叉树)
* [49. 课程表](#49-课程表)
* [50. 合并K个排序链表](#50-合并K个排序链表)
<!-- GFM-TOC -->


# 41. 找到所有数组中消失的数字

[Leetcode #448 (Easy)](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)

给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
找到所有在 [1, n] 范围之间没有出现在数组中的数字。
您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内

```html
示例:

输入:
[4,3,2,7,8,2,3,1]

输出:
[5,6]
```

- 位图法
<img src = "https://pic.leetcode-cn.com/555d915f911cc02fd8a3f80790e291544dfe4492ebbc6d58891c92694f496f07-1562932896(1).jpg">

- 元素交换
``` java
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> list=new ArrayList<>();
        if (nums==null)
            return list;
        for(int i=0;i<nums.length;)
        {
            if (nums[i]==i+1 || nums[i]==nums[nums[i]-1]  )
            {
                i++;
            }else {
                swap(nums,i,nums[i]-1);
            }
        }
        for (int i=0;i<nums.length;i++)
        {
            if (nums[i]!=i+1)
                list.add(i+1);
        }
        return list;
    }
    public void swap(int[] arr,int i,int j)
    {
        arr[i]=arr[i]+arr[j];
        arr[j]=arr[i]-arr[j];
        arr[i]=arr[i]-arr[j];
    }
}
```


# 42. 完全平方数

[Leetcode #42(Medium)](https://leetcode-cn.com/problems/perfect-squares/)
给定正整数 n，找到若干个完全平方数（比如 `1, 4, 9, 16, ...`）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

```html
示例 1:

输入: n = 12
输出: 3 
解释: 12 = 4 + 4 + 4.
示例 2:

输入: n = 13
输出: 2
解释: 13 = 4 + 9.
```

- 图论
```java
	  static class Node {
        int val;
        int step;

        public Node(int val, int step) {
            this.val = val;
            this.step = step;
        }
    }

    // 将问题转化成图论
    // 该算法在往队列里面添加节点的时候会 add 很多重复的节点，导致超时，
    // 优化办法是，加入 visited 数组，检查要 add 的数据是否已经出现过了，防止数据重复出现，从而影响图的遍历
    // 同时优化：num - i * i 表达式，只让他计算一次
    // 同时在循环体里面判断退出或返回的条件，而不是在循环体外
    public int numSquares(int n) {
        Queue<Node> queue = new LinkedList<>();
        queue.add(new Node(n, 0));
        // 其实一个真正的图的 BSF 是一定会加上 visited 数组来过滤元素的
        boolean[] visited = new boolean[n+1];
        while (!queue.isEmpty()) {
            int num = queue.peek().val;
            int step = queue.peek().step;
            queue.remove();

            for (int i = 1; ; i++) {
                int a = num - i * i;
                if (a < 0) {
                    break;
                }
                // 若 a 已经计算到 0 了，就不必再往下执行了
                if (a == 0) {
                    return step + 1;
                }
                if (!visited[a]) {
                    queue.add(new Node(a, step + 1));
                    visited[a] = true;
                }
            }
        }
        return -1;
    }
```

#43. 买卖股票的最佳时机

[Leetcode #121 (Easy)](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。

注意你不能在买入股票前卖出股票。
```html
示例 1:

输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
示例 2:

输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

- 高峰低谷问题转换 一次遍历 
```java
public class Solution {
    public int maxProfit(int prices[]) {
        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minprice)
                minprice = prices[i];
            else if (prices[i] - minprice > maxprofit)
                maxprofit = prices[i] - minprice;
        }
        return maxprofit;
    }
}
```

#44. 电话号码的字母组合

[Leetcode #17 (Medium)](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/original_images/17_telephone_keypad.png">

```html
示例:

输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```
说明:
尽管上面的答案是按字典序排列的，但是你可以任意选择答案输出的顺序。

- 回溯

```java
class Solution {
  Map<String, String> phone = new HashMap<String, String>() {{
    put("2", "abc");
    put("3", "def");
    put("4", "ghi");
    put("5", "jkl");
    put("6", "mno");
    put("7", "pqrs");
    put("8", "tuv");
    put("9", "wxyz");
  }};

  List<String> output = new ArrayList<String>();

  public void backtrack(String combination, String next_digits) {
    // if there is no more digits to check
    if (next_digits.length() == 0) {
      // the combination is done
      output.add(combination);
    }
    // if there are still digits to check
    else {
      // iterate over all letters which map 
      // the next available digit
      String digit = next_digits.substring(0, 1);
      String letters = phone.get(digit);
      for (int i = 0; i < letters.length(); i++) {
        String letter = phone.get(digit).substring(i, i + 1);
        // append the current letter to the combination
        // and proceed to the next digits
        backtrack(combination + letter, next_digits.substring(1));
      }
    }
  }

  public List<String> letterCombinations(String digits) {
    if (digits.length() != 0)
      backtrack("", digits);
    return output;
  }
}
```

# 45. 最小栈

[Leetcode #155 (Easy)](<https://leetcode-cn.com/problems/min-stack/>)

设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

- push(x) -- 将元素 x 推入栈中。
- pop() -- 删除栈顶的元素。
- top() -- 获取栈顶元素。
- getMin() -- 检索栈中的最小元素。

```html
示例:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```

- 辅助栈

```java
import java.util.Stack;

public class MinStack {

    // 数据栈
    private Stack<Integer> data;
    // 辅助栈
    private Stack<Integer> helper;

    /**
     * initialize your data structure here.
     */
    public MinStack() {
        data = new Stack<>();
        helper = new Stack<>();
    }

    // 思路 1：数据栈和辅助栈在任何时候都同步

    public void push(int x) {
        // 数据栈和辅助栈一定会增加元素
        data.add(x);
        if (helper.isEmpty() || helper.peek() >= x) {
            helper.add(x);
        } else {
            helper.add(helper.peek());
        }
    }

    public void pop() {
        // 两个栈都得 pop
        if (!data.isEmpty()) {
            helper.pop();
            data.pop();
        }
    }

    public int top() {
        if(!data.isEmpty()){
            return data.peek();
        }
        throw new RuntimeException("栈中元素为空，此操作非法");
    }

    public int getMin() {
        if(!helper.isEmpty()){
            return helper.peek();
        }
        throw new RuntimeException("栈中元素为空，此操作非法");
    }
}
```



# 46.  除法求值

[Leetcode #399 (Medium)](<https://leetcode-cn.com/problems/evaluate-division/>)

给出方程式 `A / B = k`, 其中 `A` 和 `B` 均为代表字符串的变量， `k` 是一个浮点型数字。根据已知方程式求解问题，并返回计算结果。如果结果不存在，则返回 `-1.0`。

```html
示例 :
给定 a / b = 2.0, b / c = 3.0
问题: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? 
返回 [6.0, 0.5, -1.0, 1.0, -1.0 ]
```

输入为: `vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries`(方程式，方程式结果，问题方程式)， 其中 `equations.size() == values.size()`，即方程式的长度与方程式结果长度相等（程式与结果一一对应），并且结果值均为正数。以上为方程式的描述。 返回`vector<double>`类型。

基于上述例子，输入如下：

```html
equations(方程式) = [ ["a", "b"], ["b", "c"] ],
values(方程式结果) = [2.0, 3.0],
queries(问题方程式) = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]. 
```

输入总是有效的。你可以假设除法运算中不会出现除数为0的情况，且不存在任何矛盾的结果。

- 动态规划 Floyd变种 C++代码
- [B站：[视频\]小旭讲解 LeetCode 399. Evaluate Division 并查集](https://www.bilibili.com/video/av40929397?from=search&seid=5462235911348002192)

```java
static auto x=[]()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    return 0;
}();
class Solution {
public:
    const int inf=999999999;
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        unordered_map<string,int> store;//存储对应的string对应的节点index
        int index=1;
        vector<vector<double>> graph(100, vector<double>(100, inf));
        for(int i=0;i<equations.size();++i)
        {
            if(store.find(equations[i][0])==store.end())
            {
                store[equations[i][0]]=index++;
            }
            if(store.find(equations[i][1])==store.end())
            {
                store[equations[i][1]]=index++;
            }
            graph[store[equations[i][0]]][store[equations[i][1]]]=values[i];
            graph[store[equations[i][1]]][store[equations[i][0]]]=1/values[i];
        }
        //此时1~index-1是所有节点的序号，进行Floyd算法求解答案
        
        for(int i=1;i<index;++i)
        {
            for(int j=1;j<index;++j)
            {
                for(int k=1;k<index;++k)
                {
                    if(graph[i][k]!=inf&&graph[k][j]!=inf&&graph[i][j]==inf)
                    {
                        graph[i][j]=graph[i][k]*graph[k][j];
                    }
                }
            }
        }
        
        vector<double> res(queries.size());
        for(int i=0;i<queries.size();++i)
        {
            if(store.find(queries[i][0])!=store.end()&&store.find(queries[i][1])!=store.end())
            {
                int a=store[queries[i][0]];
                int b=store[queries[i][1]];
                if(graph[a][b]!=inf)
                {
                    res[i]=graph[a][b];
                }
                else
                {
                    res[i]=-1;
                }
            }
            else
            {
                res[i]=-1;
            }
        }
        return res;
    }
};
```

# 47. 最佳买卖股票时机含冷冻期

[Leetcode #309 (Medium)](<https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/>)

给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

- 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
- 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

```html
示例:

输入: [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```

- 一个方法团灭6道股票问题
- [DP table 状态机 动态规划](<https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/solution/yi-ge-fang-fa-tuan-mie-6-dao-gu-piao-wen-ti-by-lab/>)

```java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        int dp_pre_0 = 0; // 代表 dp[i-2][0]
        for (int i = 0; i < n; i++) {
            int temp = dp_i_0;
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, dp_pre_0 - prices[i]);
            dp_pre_0 = temp;
        }
        return dp_i_0;
    }
}
```

# 48. 对称二叉树

[Leetcode #101 (Easy)](<https://leetcode-cn.com/problems/symmetric-tree/>)

给定一个二叉树，检查它是否是镜像对称的。

例如，二叉树 `[1,2,2,3,4,4,3]` 是对称的。

```html
    1
   / \
  2   2
 / \ / \
3  4 4  3
```

但是下面这个 `[1,2,2,null,3,null,3]` 则不是镜像对称的:

```html
    1
   / \
  2   2
   \   \
    3    3
```

**说明:**

如果你可以运用递归和迭代两种方法解决这个问题，会很加分。

- 递归
- 迭代

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return isMirror(root, root);
    }

    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        return (t1.val == t2.val)
            && isMirror(t1.right, t2.left)
            && isMirror(t1.left, t2.right);
    }
}
```

# 49. 课程表

[Leetcode #207 (Medium)](<https://leetcode-cn.com/problems/course-schedule/>)

现在你总共有 n 门课需要选，记为 0 到 n-1。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]

给定课程总量以及它们的先决条件，判断是否可能完成所有课程的学习？

```html
示例 1:

输入: 2, [[1,0]] 
输出: true
解释: 总共有 2 门课程。学习课程 1 之前，你需要完成课程 0。所以这是可能的。
示例 2:

输入: 2, [[1,0],[0,1]]
输出: false
解释: 总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0；并且学习课程 0 之前，你还应先完成课程 1。这是不可能的。

```

说明:

1. 输入的先决条件是由边缘列表表示的图形，而不是邻接矩阵。详情请参见图的表示法。
2. 你可以假定输入的先决条件中没有重复的边。

提示:

1. 这个问题相当于查找一个循环是否存在于有向图中。如果存在循环，则不存在拓扑排序，因此不可能选取所有课程进行学习。
2. 通过 [DFS 进行拓扑排序](<https://www.coursera.org/specializations/algorithms>) - 一个关于Coursera的精彩视频教程（21分钟），介绍拓扑排序的基本概念。
3. 拓扑排序也可以通过 BFS 完成。

- 拓扑排序 Khan算法

- 深度优先遍历

  ```java
  import java.util.ArrayList;
  import java.util.LinkedList;
  import java.util.List;
  
  /**
   * 该方法还存储了拓扑排序的结果，个人觉得这种写法很好理解，根据这个写法可以马上写出 LeetCode 第 210 题 课程表 II
   *
   * @author liwei
   * @date 18/6/24 下午12:20
   */
  public class Solution {
  
      /**
       * @param numCourses
       * @param prerequisites
       * @return
       */
      public boolean canFinish(int numCourses, int[][] prerequisites) {
          if (numCourses <= 0) {
              return false;
          }
          int plen = prerequisites.length;
          if (plen == 0) {
              return true;
          }
          int[] inDegree = new int[numCourses];
          for (int[] p : prerequisites) {
              inDegree[p[0]]++;
          }
          LinkedList<Integer> queue = new LinkedList<>();
          // 首先加入入度为 0 的结点
          for (int i = 0; i < numCourses; i++) {
              if (inDegree[i] == 0) {
                  queue.addLast(i);
              }
          }
          // 拓扑排序的结果
          List<Integer> res = new ArrayList<>();
          while (!queue.isEmpty()) {
              Integer num = queue.removeFirst();
              res.add(num);
              // 把邻边全部遍历一下
              for (int[] p : prerequisites) {
                  if (p[1] == num) {
                      inDegree[p[0]]--;
                      if (inDegree[p[0]] == 0) {
                          queue.addLast(p[0]);
                      }
                  }
              }
          }
          // System.out.println("拓扑排序结果：");
          // System.out.println(res);
          return res.size() == numCourses;
      }
  }
  ```

# 50. 合并K个排序链表

[Leetcode #23 (Difficult)](<https://leetcode-cn.com/problems/merge-k-sorted-lists/>)

合并 *k* 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。

```html
示例:

输入:
[
  1->4->5,
  1->3->4,
  2->6
]
输出: 1->1->2->3->4->4->5->6
```

- 暴力
- 逐一比较
- 优先队列优化2
- 逐一两两合并
- 分治

[贪心算法、优先队列](<https://leetcode-cn.com/problems/merge-k-sorted-lists/solution/tan-xin-suan-fa-you-xian-dui-lie-fen-zhi-fa-python/>)

[分治法、递归](https://leetcode-cn.com/problems/merge-k-sorted-lists/solution/javafen-zhi-fa-di-gui-onlogk-by-heator/)

```java
class Solution {
    
    /**
     * 合并 k 个排序链表，返回合并后的排序链表。
     * 两两合并,分治法,O(nlogk)
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        int len=lists.length;
        if(len<1) return null;
        if (len==1) return lists[0];
        //分解list1
        ListNode[] ps11 = new ListNode[len/2],ps12 = new ListNode[len/2];
        System.arraycopy(lists,0,ps11,0,len/2);
        System.arraycopy(lists,len/2,ps12,0,len/2);
        //判断lists[]长度为奇数还是偶数
        return len%2==0?mergeList(ps11,ps12):mergeListNode(lists[len-1],mergeList(ps11,ps12));
    }

    /**
     * 分治法,递归,O(nlogk)
     * @param list1
     * @param list2
     */
    private static ListNode mergeList(ListNode[] list1,ListNode[] list2){
        int len1=list1.length,len2=list2.length;
        if (len1<=1&&len2<=1) {
            return mergeListNode(list1[0],list2[0]);
        }else {
            //分解list1
            ListNode[] ps11 = new ListNode[len1/2],ps12 = new ListNode[len1/2];
            System.arraycopy(list1,0,ps11,0,len1/2);
            System.arraycopy(list1,len1/2,ps12,0,len1/2);
            //分解list2
            ListNode[] ps21 = new ListNode[len2/2],ps22 = new ListNode[len2/2];
            System.arraycopy(list2,0,ps21,0,len2/2);
            System.arraycopy(list2,len2/2,ps22,0,len2/2);
            //如果两个list[]长度均为奇数
            if (len1%2!=0&&len2%2!=0){
                return mergeListNode(
                        mergeListNode(list1[len1-1],list2[len2-1]),
                        mergeListNode(mergeList(ps11,ps12),mergeList(ps21,ps22))
                );
            }else if (len1%2!=0){
                //如果list1[]长度为奇数
                return mergeListNode(
                        list1[len1-1],
                        mergeListNode(mergeList(ps11,ps12),mergeList(ps21,ps22))
                        );
            }else if(len2%2!=0) {
                //如果list2[]长度为奇数
                return mergeListNode(
                        mergeListNode(mergeList(ps11,ps12),mergeList(ps21,ps22)),
                        list2[len2-1]
                        );
            }else {
                //如果均为偶数
                return mergeListNode(mergeList(ps11,ps12),mergeList(ps21,ps22));
            }
        }
    }

    /**
     * 合并两个链表
     * @param l1
     * @param l2
     * @return
     */
    private static ListNode mergeListNode(ListNode l1, ListNode l2){
        ListNode dummyhead=new ListNode(0);
        merge(dummyhead,l1,l2);
        return dummyhead.next;
    }

    /**
     * 递归合并
     * @param head
     * @param l1
     * @param l2
     * @return
     */
    private static void merge(ListNode head, ListNode l1, ListNode l2) {
        if (l1 != null && l2 != null) {
            //把head.next和l1.next放到下一轮递归
            if (l1.val <= l2.val) {
                head.next = l1;
                ListNode p = l1.next;
                l1.next = null;
                merge(head.next, p, l2);
            } else {
                //把head.next和l2.next放到下一轮递归
                head.next = l2;
                ListNode p = l2.next;
                l2.next = null;
                merge(head.next, l1, p);
            }
        }
        if (l1 == null && l2 == null) ;
        if (l1 == null && l2 != null) {
            //把head.next和l2.next放到下一轮递归
            head.next = l2;
            ListNode p = l2.next;
            l2.next = null;
            merge(head.next, l1, p);

        }
        if (l1 != null && l2 == null) {
            //把head.next和l1.next放到下一轮递归
            head.next = l1;
            ListNode p = l1.next;
            l1.next = null;
            merge(head.next, p, l2);
        }
    }
}
```

