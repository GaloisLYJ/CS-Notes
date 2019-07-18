<!-- GFM-TOC -->

* [11. 反转链表](#11-反转链表)
* [12. 旋转图像](#12-旋转图像)
* [13. 二叉树展开为链表](#13-二叉树展开为链表)
* [14. 根据身高重建队列](#14-根据升高重建队列)
* [15. 只出现一次的数字](#15-只出现一次的数字)
* [16. 翻转二叉树](#6-翻转二叉树)
* [17. 全排列](#7-全排列)
* [18. 二叉树的最大深度](#8-二叉树的最大深度)
* [19. 二叉树的中序遍历](#9-二叉树的中序遍历)]
* [20. 组合总和](#10-组合总和)
  <!-- GFM-TOC -->


# 11. 反转链表

[Leetcode #206 (Easy)](<https://leetcode-cn.com/problems/reverse-linked-list/>)

反转一个单链表。

```html
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

**进阶:**
你可以迭代或递归地反转链表。你能否用两种方法解决这道题？



假设存在链表 `1 → 2 → 3 → Ø`，我们想要把它改成 `Ø ← 1 ← 2 ← 3`。

在遍历列表时，将当前节点的 `next` 指针改为指向前一个元素。由于节点没有引用其上一个节点，因此必须事先存储其前一个元素。在更改引用之前，还需要另一个指针来存储下一个节点。不要忘记在最后返回新的头引用！

链表知识还需要画图明细。

```java
public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;
    while (curr != null) {
        ListNode nextTemp = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextTemp;
    }
    return prev;
}
```

# 12. 旋转图像

[Leetcode #48 (Medium)](https://leetcode-cn.com/problems/rotate-image/)
给定一个 n × n 的二维矩阵表示一个图像。

将图像顺时针旋转 90 度。

说明：

你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

```html
示例 1:

给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
示例 2:

给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
```
最直接的想法是先转置矩阵，然后翻转每一行。这个简单的方法已经能达到最优的时间复杂度O(N^2)。

- 时间复杂度：O(N^2)*O*(*N*2).
- 空间复杂度：O(1)*O*(1) 由于旋转操作是 *就地* 完成的。

为啥我的操作不行，因为题目没理解到位，只做了转置。
```java
public void rotate(int[][] matrix) {
    int n = matrix.length;

    // transpose matrix
    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {
        int tmp = matrix[j][i];
        matrix[j][i] = matrix[i][j];
        matrix[i][j] = tmp;
      }
    }
    // reverse each row
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n / 2; j++) {
        int tmp = matrix[i][j];
        matrix[i][j] = matrix[i][n - j - 1];
        matrix[i][n - j - 1] = tmp;
      }
    }
}
```

# 13. 二叉树展开为链表

[Leetcode #114 (Medium)](<https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/>)
给定一个二叉树，原地将它展开为链表。

```html
例如，给定二叉树

    1
   / \
  2   5
 / \   \
3   4   6
将其展开为：

1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

传统前序遍历的变形，还有morris解法可研究。暂时理解不好。(看代码理解了理解不好)
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
    // 增加全局last节点
    TreeNode last = null;
    public void flatten(TreeNode root) {
        if (root == null) return;
        // 前序：注意更新last节点，包括更新左右子树
        if (last != null) {
            last.left = null;
            last.right = root;
        }
        last = root;
        // 前序：注意备份右子节点，规避下一节点篡改
        TreeNode copyRight = root.right;
        flatten(root.left);
        flatten(copyRight);
    }
}
```
# 14. 根据身高重建队列

[Leetcode #406 (Medium)](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)
假设有打乱顺序的一群人站成一个队列。 每个人由一个整数对(h, k)表示，其中h是这个人的身高，k是排在这个人前面且身高大于或等于h的人数。 编写一个算法来重建这个队列。
注意：
总人数少于1100人。

```html
输入:
[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

输出:
[[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
```
贪心算法，例子
```java
	  public int[][] reconstructQueue(int[][] people) {
        if (0 == people.length || 0 == people[0].length) return new int[0][0];
        //按照身高降序 K升序排序 
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] == o2[0] ? o1[1] - o2[1] : o2[0] - o1[0];
            }
        });
        List<int[]> list = new ArrayList<>();
        //K值定义为 排在h前面且身高大于或等于h的人数
        //因为从身高降序开始插入，此时所有人身高都大于等于h
        //因此K值即为需要插入的位置
        for (int i=0;i<people.length;i++) {
            list.add(people[i]);
        }
        int[] temp = new int[2];
        for(int i=0;i<list.size();i++){
            temp = list.get(i);
            list.remove(i);
            list.add(temp[1],temp);
        }
        return list.toArray(new int[list.size()][]);
    }
```

# 15. 只出现一次的数字

[Leetcode #136 (Easy)](https://leetcode-cn.com/problems/single-number/)
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
- 你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
注意：
```html
示例 1:

输入: [2,2,1]
输出: 1
示例 2:

输入: [4,1,2,1,2]
输出: 4
```
这题居然有四种解法，异或又是什么操作？好好学
```java
    public int singleNumber(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (Integer i : nums) {
            Integer count = map.get(i);
            count = count == null ? 1 : ++count;
            map.put(i, count);
        }
        for (Integer i : map.keySet()) {
            Integer count = map.get(i);
            if (count == 1) {
                return i;
            }
        }
        return -1; // can't find it.
    }
```

# 16. 除自身以外数组的乘积

[Leetcode #238  (Medium)](https://leetcode-cn.com/problems/product-of-array-except-self/)
给定长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
```html
输入: [1,2,3,4]
输出: [24,12,8,6]
```
说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。
进阶：
你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）


```java
    
    
````

# 17. 全排列

[Leetcode #46  (Medium)](https://leetcode-cn.com/problems/permutations/)
给定一个没有重复数字的序列，返回其所有可能的全排列。

```html
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```
```html
回溯法，暂时理解不透，记录之
```

回溯法 是一种通过探索所有可能的候选解来找出所有的解的算法。如果候选解被确认 不是 一个解的话（或者至少不是 最后一个 解），回溯算法会通过在上一步进行一些变化抛弃该解，即 回溯 并且再次尝试。

这里有一个回溯函数，使用第一个整数的索引作为参数 backtrack(first)。

* 如果第一个整数有索引 n，意味着当前排列已完成。
* 遍历索引 first 到索引 n - 1 的所有整数。Iterate over the integers from index first to index n - 1.
  - 在排列中放置第 i 个整数， 即 swap(nums[first], nums[i]).
  - 继续生成从第 i 个整数开始的所有排列: backtrack(first + 1).
  - 现在回溯，即通过 swap(nums[first], nums[i]) 还原.
<div align="center"> <img src="img/Permutations.png" width=""/> </div><br>
```java
import java.util.Collections;
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        // init output list2
        List<List<Integer>> output = new LinkedList();

        // convert nums into list since the output is a list of lists
        ArrayList<Integer> nums_lst = new ArrayList<Integer>();
        for (int num : nums)
          nums_lst.add(num);

        int n = nums.length;
        backtrack(n, nums_lst, output, 0);
        return output;
    }
    
    public void backtrack(int n,ArrayList<Integer> nums,List<List<Integer>> output,int first){
        // if all integers are used up
        if (first == n)
          output.add(new ArrayList<Integer>(nums));
        for (int i = first; i < n; i++) {
          // place i-th integer first 
          // in the current permutation
          Collections.swap(nums, first, i);
          // use next integers to complete the permutations
          backtrack(n, nums, output, first + 1);
          // backtrack
          Collections.swap(nums, first, i);
        }
    }
}
```

# 8. 二叉树的最大深度

[Leetcode #104  (Easy)](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：
给定二叉树 `[3,9,20,null,null,15,7]，`

```html
    3
   / \
  9  20
    /  \
   15   7
```

返回它的最大深度 3 。

递归

```java
    public int maxDepth(TreeNode root) {
        if(root == null){
            return 0;
        }else{
            int left_height = maxDepth(root.left);
            int right_height = maxDepth(root.right);
            return java.lang.Math.max(left_height,right_height) + 1;
        }
    }
```



# 9. 二叉树的中序遍历
[Leetcode #94  (Easy)](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

给定一个二叉树，返回它的*中序* 遍历。

```html
输入: [1,null,2,3]
   1
    \
     2
    /
   3

输出: [1,3,2]
```

**进阶:** 递归算法很简单，你可以通过迭代算法完成吗？



必会基础题，在之前牛客算法班视频刷过，结果忘了，不过好处是有了算法任务清单的诞生，算法任务清单.md

```java
    public List < Integer > inorderTraversal(TreeNode root) {
        List < Integer > res = new ArrayList < > ();
        helper(root, res);
        return res;
    }

    public void helper(TreeNode root, List < Integer > res) {
        if (root != null) {
            if (root.left != null) {
                helper(root.left, res);
            }
            res.add(root.val);
            if (root.right != null) {
                helper(root.right, res);
            }
        }
    }
```

#10. 组合总和
[Leetcode #39  (Medium)](https://leetcode-cn.com/problems/combination-sum/)

给定一个无重复元素的数组 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。

`candidates` 中的数字可以无限制重复被选取。
说明：

- 所有数字（包括 `target`）都是正整数。
- 解集不能包含重复的组合。 

```html
示例 1:

输入: candidates = [2,3,6,7], target = 7,
所求解集为:
[
  [7],
  [2,2,3]
]
示例 2:

输入: candidates = [2,3,5], target = 8,
所求解集为:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```



回溯法思想汇总，与分析问题的思路示范。

- [回溯算法 + 剪枝 python、java](https://leetcode-cn.com/problems/combination-sum/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-2/)
- [学一套回溯算法 走天下](https://leetcode-cn.com/problems/combination-sum/solution/xue-yi-tao-zou-tian-xia-hui-su-suan-fa-by-powcai/)

```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;

public class Solution {

    private List<List<Integer>> res = new ArrayList<>();
    private int[] candidates;
    private int len;

    private void findCombinationSum(int residue, int start, Stack<Integer> pre) {
        if (residue == 0) {
            res.add(new ArrayList<>(pre));
            return;
        }
        // 优化添加的代码2：在循环的时候做判断，尽量避免系统栈的深度
        // residue - candidates[i] 表示下一轮的剩余，如果下一轮的剩余都小于 0 ，就没有必要进行后面的循环了
        // 这一点基于原始数组是排序数组的前提，因为如果计算后面的剩余，只会越来越小
        for (int i = start; i < len && residue - candidates[i] >= 0; i++) {
            pre.add(candidates[i]);
            // 【关键】因为元素可以重复使用，这里递归传递下去的是 i 而不是 i + 1
            findCombinationSum(residue - candidates[i], i, pre);
            pre.pop();
        }
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        int len = candidates.length;
        if (len == 0) {
            return res;
        }
        // 优化添加的代码1：先对数组排序，可以提前终止判断
        Arrays.sort(candidates);
        this.len = len;
        this.candidates = candidates;
        findCombinationSum(target, 0, new Stack<>());
        return res;
    }

    public static void main(String[] args) {
        int[] candidates = {2, 3, 6, 7};
        int target = 7;
        Solution solution = new Solution();
        List<List<Integer>> combinationSum = solution.combinationSum(candidates, target);
        System.out.println(combinationSum);
    }
}
```