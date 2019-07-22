<!-- GFM-TOC -->
* [41. 找到所有数组中消失的数字](#41-找到所有数组中消失的数字)

* [42. 完全平方数](#42-完全平方数)
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
