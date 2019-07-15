<!-- GFM-TOC -->
* [1. 子集](#1-子集)
* [2. 比特位计数](#2-比特位计数)
* [3. 合并二叉树](#3-合并二叉树)
* [4. 括号生成](#4-括号生成)
* [5. 汉明距离](#5-汉明距离)
<!-- GFM-TOC -->


# 1. 子集

[Leetcode #78 (Medium)](https://leetcode-cn.com/problems/subsets/)

给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。

```html
输入: nums = [1,2,3]
输出:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
```

循环枚举，逐个枚举，空集的幂集只有空集，原集每个元素加上新的元素就是新的子集
```java
    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        res.add(new ArrayList<Integer>());
        for (Integer n : nums) {
            int size = res.size();
            for (int i = 0; i < size; i++) {
                List<Integer> newSub = new ArrayList<Integer>(res.get(i));
                newSub.add(n);
                res.add(newSub);
            }
        }
        return res;
    }
```

# 2. 比特位计数

[Leetcode #338 (Medium)](https://leetcode-cn.com/problems/counting-bits/)
给定一个非负整数 num。对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。

进阶:
-给出时间复杂度为O(n*sizeof(integer))的解答非常容易。但你可以在线性时间O(n)内用一趟扫描做到吗？
-要求算法的空间复杂度为O(n)。
-你能进一步完善解法吗？要求在C++或任何其他语言中不使用任何内置函数（如 C++ 中的 __builtin_popcount）来执行此操作。
```html
示例 1:

输入: 2
输出: [0,1,1]
示例 2:

输入: 5
输出: [0,1,1,2,1,2]
```
一个大于零的整数，每次和比自己小1的数，做一次与运算，就减1，同时少一位1
```java
    public int[] countBits(int num) {
        int[] ans = new int[num + 1];
        for(int i=0;i<=num;i++){
            ans[i] = popcount(i);
        }
        return ans;
    }

    private int popcount(int x){
        int count = 0;
        while(x!=0){
            x &= x-1;
            count++;
        }
        return count;
    }
```

# 3. 合并二叉树

[Leetcode #617 (Easy)](https://leetcode-cn.com/problems/merge-two-binary-trees/)
给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。

你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

注意: 合并必须从两个树的根节点开始。
```html
输入: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
输出: 
合并后的树:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7
```

递归，注意边界条件，和`new TreeNode(t1.val + t2.val);`内值的构造
```java
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if(t1 == null){
            return t2;
        }
        if(t2 == null){
            return t1;
        }
        TreeNode result = new TreeNode(t1.val + t2.val);
        result.left = mergeTrees(t1.left,t2.left);
        result.right = mergeTrees(t1.right,t2.right);
        return result;
    }
```
# 4. 括号生成

[Leetcode #22 (Medium)](https://leetcode-cn.com/problems/generate-parentheses/)
给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
例如，给出 n = 3，生成结果为：
```html
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```
较为复杂，暂未理解好
```java
public List<String> generateParenthesis(int n) {
        List<String> combinations = new ArrayList<String>();
        generateAll(new char[2*n],0,combinations);
        return combinations;
    }
    
    public void generateAll(char[] current,int pos,List<String> result){
        if(pos == current.length){
            if(valid(current)){
                result.add(new String(current));
            }
        }else{
            current[pos] = '(';
            generateAll(current,pos+1,result);
            current[pos] = ')';
            generateAll(current,pos+1,result);
        }
    }
    
    public boolean valid(char[] current){
        int balance = 0;
        for(char c:current){
            if(c == '('){
                balance++;
            }else{
                balance--;
            }
            if(balance<0) return false;
        }
        return (balance == 0);
    }
```

# 5. 汉明距离

[Leetcode #461 (Easy)](https://leetcode-cn.com/problems/hamming-distance/)
两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
给出两个整数 x 和 y，计算它们之间的汉明距离。
注意：
0 ≤ x, y < 231.
```html
输入: x = 1, y = 4

输出: 2

解释:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑

上面的箭头指出了对应二进制位不同的位置。
```
按位与 `&` 是只有都是1的时候才是1
按位或 `|` 是只有都是0的时候才是0
按位异或 `^` 相同的时候是相同，不同的时候是1
```java
    public int hammingDistance(int x, int y) {
        int result = x^y;
        int count = 0;
        while(result!=0){
            result = result & result-1;
            count++;
        }
        return count;
    }
```