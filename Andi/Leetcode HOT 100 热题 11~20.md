<!-- GFM-TOC -->
* [11. 反转链表](#11-反转链表)
* [12. 旋转图像](#12-旋转图像)
* [13. 二叉树展开为链表](#13-二叉树展开为链表)
* [14. 根据身高重建队列](#14-根据升高重建队列)
* [15. 只出现一次的数字](#15-只出现一次的数字)
* [16. 翻转二叉树](#16-翻转二叉树)
* [17. 全排列](#17-全排列)
* [18. 不同的二叉搜索树](#18-不同的二叉搜索树)
* [19. 排序链表](#19-排序链表)
* [20. 寻找重复数](#20-寻找重复数)
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

分治法思想 
- 乘积 = 当前数左边的乘积 * 当前数右边的乘积
```java
    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        int k = 1;
        for(int i = 0; i < res.length; i++){
            res[i] = k;
            k = k * nums[i]; // 此时数组存储的是除去当前元素左边的元素乘积
        }
        k = 1;
        for(int i = res.length - 1; i >= 0; i--){
            res[i] *= k; // k为该数右边的乘积。
            k *= nums[i]; // 此时数组等于左边的 * 该数右边的。
        }
        return res;
    }
````

# 17. 最小路径和

[Leetcode #64  (Medium)](https://leetcode-cn.com/problems/minimum-path-sum/)
给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。
```html
输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。
```

- 暴力递归(超出时间限制)
- 动态规划
```java
    public int calculate(int[][] grid, int i, int j) {
        if (i == grid.length || j == grid[0].length) return Integer.MAX_VALUE;
        if (i == grid.length - 1 && j == grid[0].length - 1) return grid[i][j];
        return grid[i][j] + Math.min(calculate(grid, i + 1, j), calculate(grid, i, j + 1));
    }
    public int minPathSum(int[][] grid) {
        return calculate(grid, 0, 0);
    }
```

# 18. 二叉树的最大深度

[Leetcode #96  (Medium)](https://leetcode-cn.com/problems/unique-binary-search-trees/)
给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？
```html
输入: 3
输出: 5
解释:
给定 n = 3, 一共有 5 种不同结构的二叉搜索树:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

- 动态规划 迟早要攻克你
```java
  public int numTrees(int n) {
    int[] G = new int[n + 1];
    G[0] = 1;
    G[1] = 1;

    for (int i = 2; i <= n; ++i) {
      for (int j = 1; j <= i; ++j) {
        G[i] += G[j - 1] * G[i - j];
      }
    }
    return G[n];
  }

```



# 19. 排序链表

[Leetcode #148  (Medium)](https://leetcode-cn.com/problems/sort-list/)
在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序。

```html
示例 1:

输入: 4->2->1->3
输出: 1->2->3->4
示例 2:

输入: -1->5->3->4->0
输出: -1->0->3->4->5
```

- [链表归并排序](https://leetcode-cn.com/problems/sort-list/solution/sort-list-gui-bing-pai-xu-lian-biao-by-jyd/)
```java
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode fast = head.next, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode tmp = slow.next;
        slow.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(tmp);
        ListNode h = new ListNode(0);
        ListNode res = h;
        while (left != null && right != null) {
            if (left.val < right.val) {
                h.next = left;
                left = left.next;
            } else {
                h.next = right;
                right = right.next;
            }
            h = h.next;
        }
        h.next = left != null ? left : right;
        return res.next;
    }
```

#20. 寻找重复数

[Leetcode #287  (Medium)](https://leetcode-cn.com/problems/find-the-duplicate-number/)
给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。
```html
示例 1:

输入: [1,3,4,2,2]
输出: 2
示例 2:

输入: [3,1,3,4,2]
输出: 3
```
说明：

1. 不能更改原数组（假设数组是只读的）。
2. 只能使用额外的 O(1) 的空间。
3. 时间复杂度小于 O(n2) 。
4. 数组中只有一个重复的数字，但它可能不止重复出现一次。

- 抽屉原理
- 面试真实
- 排序思想
有最好的方法，面试官竟然不希望有人直接给出，会被认为是有备而来，不是真实水平。
```java
	 public int findDuplicate(int[] nums) {
        Arrays.sort(nums);
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i-1]) {
                return nums[i];
            }
        }

        return -1;
    }
```