# Java重写：灵茶山艾府——基础算法精讲：

<font color=#f00 size= 7>**做满500道题再谈天赋**</font>

[TOC]



# <font color=#f00 size= 7>双指针</font>

## 一.相向双指针：

### 167.两数之和：

主要算法原理，对于有序数列，首尾相加的结果与target进行比较，如果比target小左指针右移，比target大右指针左移

~~~java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int left = 0;
        int right = numbers.length - 1;
        while(left < right){
            if(numbers[left] + numbers[right] < target){
                left++;
            }else if(numbers[left] + numbers[right] > target){
                right--;
            }else{
                return new int[]{left + 1, right + 1};
            }
        }
        return new int[]{-1, -1};
    }
}
~~~

### 15.三数之和：

![image-20240405191800184](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405191800184.png)

此题是在上一题中升级出来的，可以将其中一个数看成target的序列

~~~java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i < n - 2; i++) {
            //开始进行剪枝操作
            //如果当前值和下一个的值一样直接进入下一次循环
            if (i > 0 && nums[i] == nums[i - 1]){
                continue;
            }
            //如果当前数大于0，则后面的数肯定大于0，不用再继续判断了
            if(nums[i] + nums[i + 1] + nums[i + 2] > 0){
                break;
            }
            //如果当前数小于0，则后面的数肯定小于0，不用再继续判断了
            if(nums[i] + nums[n-2] + nums[n-1] < 0){
                continue;
            }
            int j = i + 1, k = n - 1;
            while(j < k){
                if(nums[i] + nums[j] + nums[k] == 0){
                    res.add(Arrays.asList(nums[i],nums[j],nums[k]));
                    j++;
                    while(j < k && nums[j] == nums[j - 1]){
                        j++;
                    }
                    k--;
                    while(j < k && nums[k] == nums[k + 1]){
                        k--;
                    }
                }else if(nums[i] + nums[j] + nums[k] > 0){
                    k--;
                }else{
                    j++;
                }
            }
        }
        return res;
    }
}
~~~



## 二.相向双指针

### 11.盛最多水的容器（ak）

贪心算法，从最外侧两个板向中间移动，如果想要容量增大，则要保证高度比原来的高，由于水桶的容量是靠短板控制，所以贪心规则是不断将短板向中间移动，直到两块板相遇，记录过程中的所有的容量，求出最大值。

~~~java
class Solution {
    public int maxArea(int[] height) {
        /**
        使用贪心算法：
        每次都将矮的板向中间移动一格，维护一格全局变量记录最大值
         */
         int n = height.length;
         int left = 0;
         int right = n -1 ;
         int max = 0;
         int cup = 0;
         while(left < right){
             cup = Math.min(height[left],height[right]) * (right - left);
             
             if(height[left]> height[right]){
                 right--;
             }else{
                 left++;
             }
             if(cup > max){
                 max = cup;
             }
         }
         return max;
    }
}
~~~

### 42.接雨水问题（ak）

~~~java
class Solution {
    public int trap(int[] height) {
        /**
        维护两个数组，一个数组代表从左往右每个位置的右边的最大高度；一个数组表示从右往左每个位置右边的最大高度，
        之后每个位置能存储的最大水量，就是两个数组的公共位置的最小值-这个位置的高度
         */
         int n = height.length;
         //首先维护从左向右的数组
         int[] leftRight = new int[n];
         int maxLeft = 0;
         for(int i = 0; i < n; i++){
             if(height[i] > maxLeft){
                 leftRight[i] = height[i];
                 maxLeft = height[i];
             }else{
                 leftRight[i] = maxLeft;
             }
         }
         //再维护从右向左的数组
         int[] rightLeft = new int[n];
         int maxRight = 0;
         for(int i = n-1; i >= 0; i--){
             if(height[i] > maxRight){
                 rightLeft[i] = height[i];
                 maxRight = height[i];
             }else{
                 rightLeft[i] = maxRight;
             }
         }
         //计算总共存储的雨水
         int res = 0;
         int temp = 0;
         for(int i = 0; i < n ; i++){
             temp = Math.min(leftRight[i],rightLeft[i]) - height[i];
             res += temp;
         }
         return res;

    }
}
~~~

#### 空间优化：

将两个指针进行合并

~~~JAVA

~~~

## 三.同向双指针——滑动窗口题

### 209.长度最小的子数组

![image-20240405192003287](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405192003287.png)

~~~JAVA
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        /**
        思路：滑动窗口，维护一个全局变量，用来存储最小的数组长度
         */
         int n = nums.length;
         int minLength = n + 1;
         int left = 0;
         int right = 0;
         int sum = 0;
         while(right >= left && right < n){
             sum += nums[right];
             while(sum >= target){
                 minLength = Math.min(minLength, right - left + 1);
                 sum -= nums[left];
                 left++;
             }
             right++;
         }
         //如果minLength的长度<=n，证明存在sum>=target情况，此时返回minLength
         return minLength<=n?minLength:0;
         
    }
}
~~~

### 713.乘积小于k的子数组

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回子数组内所有元素的乘积严格==小于== `k` 的连续子数组的数目。

~~~java
class Solution {
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        /**
        思路：滑动窗口，维护一个全局变量，用来存储最小的数组长度
         */
        if(k <= 1){
            return 0;
        }
        int res = 0;
        int left = 0;
        int total = 1;
        int n = nums.length;
        for(int right=0 ; right < n;right++){
            total*= nums[right];
            while(total>=k){
                total /= nums[left];
                left += 1;
            }
            //这一步是计算子数组的数目，因为如果{1,2,3}满足条件，那么{2,3}和{3}也是满足条件的，注意在计算子数组的数目时需要固定一端，这里固定的是右侧
            res += right -left + 1;
        }
        return res;
    }
}
~~~

### 3.无重复的最长子串：

给定一个字符串 `s` ，请你找出其中==不含有==重复字符的 **最长子串** 的长度。

~~~java
class Solution {
    public int lengthOfLongestSubstring(String S) {
        char[] s = S.toCharArray(); // 转换成 char[] 加快效率（忽略带来的空间消耗）
    int left = 0, ans = 0, n = s.length;
    HashMap<Character, Integer> counter = new HashMap<>();
    for (int right = 0; right < n; right++) {
        char c = s[right];
        if (counter.containsKey(c)) {
            counter.put(c, counter.get(c) + 1);
        } else {
            counter.put(c, 1);
        }
        while (counter.get(c) > 1) {
            counter.put(s[left], counter.get(s[left]) - 1);
            left++;
        }
        
        ans = Math.max(ans, right - left + 1);
    }
    return ans;
    }
}
~~~

~~~java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        char[] S = s.toCharArray();
        int n = S.length, left = 0, res = 0;
        HashMap<Character, Integer> Counter = new HashMap<>();
        // 利用滑动窗口，右端往右走并不断判断是否出现重复字符，如果出现重复字符记录当前滑动窗口长度
        // 利用循环将左边往右走直到没有重复的字符
        // 继续将右端往右移动，直到出现重复的字符，记录滑动窗口的长度

        for (int right = 0; right < n; right++) {
            // 判断当前right指针指向的字符是否在Map中
            char c = S[right];
            if (Counter.containsKey(c)) {
                Counter.put(c, Counter.get(c) + 1);
            } else {
                Counter.put(c, 1);
            }
            while (Counter.get(c) > 1) {
                Counter.put(S[left], Counter.get(S[left]) - 1);
                left++;
            }
            res = Math.max(res, right - left + 1);
        }
        return res;
    }
}
~~~

# <font color=#f00 size= 7>二分查找</font>

## 四.二分查找

### 34.在排序数组中查找元素的第一个和最后一个位置

![image-20240217152608894](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240217152608894.png)

~~~java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int start = lowerBound(nums, target); // 选择其中一种写法即可
        if (start == nums.length || nums[start] != target)
            return new int[]{-1, -1};
        // 如果 start 存在，那么 end 必定存在
        int end = lowerBound(nums, target + 1) - 1;
        return new int[]{start, end};
    }
    public int lowerBound(int[] nums, int target) {
        /**
         * 数组nums非递减顺序排列
         * 使用闭区间书写
         */
        int left = 0;
        int right = nums.length - 1;
        while(left <= right){
            int mid = (left + right) / 2;
            if(nums[mid] < target){
                left = mid + 1;
            }else{
                // 这个地方为什么是mid-1，设想只有一个元素的话，如果right=mid则一直没变，就成了死循环，所以这个地方一定是mid-1
                //另一种思路，只要是在left和right之间的元素就应该是不确定的部分，不能包含已经确定的mid所以要更新成mid+1与mid-1
                right = mid - 1;
            }
        }
        return left;
    }
}
~~~

## 五.二分查找

### 162.寻找峰值：

峰值元素是指其值严格大于左右相邻值的元素。

给你一个整数数组 `nums`，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 **任何一个峰值** 所在位置即可。

你可以假设 `nums[-1] = nums[n] = -∞` 。

你必须实现时间复杂度为 `O(log n)` 的算法来解决此问题。

~~~java
class Solution {
    public int findPeakElement(int[] nums) {
        int i = 0, j = nums.length - 2;
        while(i <= j){
            int mid = (i + j) >>> 1;
            if(nums[mid] < nums[mid + 1]){
                i = mid + 1;
            }else{
                j = mid - 1;
            }
        }
        return i;
    }
}
~~~

### 153.寻找旋转排序数组中的端点

![image-20240405195026421](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405195026421.png)

~~~java
class Solution {
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (nums[right] < nums[mid]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }
}
~~~

### 154.<span style="background-color:#00ffff;">寻找旋转排序数组中的最小值II</span>

![image-20240405195045207](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405195045207.png)

~~~java
class Solution {
    public int findMin(int[] nums) {
        //这道题相较于153的难点就在于，存在mid和right相等的情况，此时不一定是右指针就是最小值
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) >>> 1;
            if (nums[right] < nums[mid]) {
                left = mid + 1;
            }else if(nums[mid] > nums[right]) {
                right = mid - 1;
            }else{
                right--;
            }
        }
        return nums[left];
    }
}
~~~

# <font color=#f00 size= 7>链表</font>

## 六.反转链表

### 206.翻转列表：

![image-20240217194632954](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240217194632954.png)

![image-20240217165214203](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240217165214203.png)

思路将链表的后节点的指针指向上一节点，将原来的首节点指向空，注意应设置三个临时变量来记录变化过程中的前后节点以及next指针。

~~~java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode pre = null, cur = head;
        while (cur != null) {
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/reverse-linked-list/solutions/1992225/you-xie-cuo-liao-yi-ge-shi-pin-jiang-tou-o5zy/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



~~~java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        //方法二：在原有的链表上进行更改
        ListNode pre = head;
        ListNode next = new ListNode();
        ListNode cur = head.next;
        while(cur != null){
            cur = pre.next;
            next = cur.next;
            cur.next = pre;
            cur = next;
        }

    }
}
~~~

### 92.反转链表II

![image-20240217194555803](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240217194555803.png)

~~~java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        //首先确定需要进行反转的链表节点
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode p0 = dummy;
        for(int i = 0; i< left - 1; i++){
            p0 = p0.next;
        }
        ListNode cur = p0.next;
        ListNode pre = null;
        for(int i = 0; i < right - left + 1; i++){
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        //最后将反转后的链表段连接到链表的原头尾
        
        p0.next.next = cur;
        p0.next = pre;
        
        return dummy.next;
    }
}
~~~

### 25.K个一组翻转链表：

![image-20240217194534154](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240217194534154.png)

~~~java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 * int val;
 * ListNode next;
 * ListNode() {}
 * ListNode(int val) { this.val = val; }
 * ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 * 这题相较于上一题就是相当于每次进行翻转链表之后需要将p0节点移动到下一次翻转链表长度的前一个节点
 */
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        // 获得链表的长度，判断剩余长度是否大于k能够进行翻转
        int n = 0;
        ListNode cur = head;
        while (cur != null) {
            n++;
            cur = cur.next;
        }

        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode p0 = dummy;
        // 通过循环不断对于链表进行翻转
        while (n >= k) {
            n -= k;

            cur = p0.next;
            ListNode pre = null;
            for (int i = 0; i < k; i++) {
                ListNode nxt = cur.next;
                cur.next = pre;
                pre = cur;
                cur = nxt;
            }
            ListNode nxt = p0.next;
            p0.next.next = cur;
            p0.next = pre;
            p0 = nxt;

        }

        return dummy.next;

    }
}
~~~

## 七.快慢指针、环形链表

### 876.链表的中间结点

~~~java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode middleNode(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
}
~~~

### 141.环形链表I

~~~java
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
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow){
                return true;
            }
        }
        return false;
    }
}
~~~

### 142.环形链表II

![image-20240405202940666](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405202940666.png)

<font color=#00f>**环形链表当快慢指针相遇后，将快指针重新放到头节点，慢指针保持在原来运动到的地方，两个指针同时以一次一步的速度再次相遇必然是在入环处。**</font>

~~~java
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
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow){
                fast = head;
                while(fast != slow){
                    fast = fast.next;
                    slow = slow.next;
                }
                return slow;
                
            }
        }
        return null;
        
    }
}
~~~

### 143.链表重排

![image-20240405202957489](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405202957489.png)

![image-20240217201551762](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240217201551762.png)

![image-20240217201619166](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240217201619166.png)

~~~java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public void reorderList(ListNode head) {
        //说明为什么这个地方调用函数的时候需要使用this，通过使用this表明是调用Solution类中的声明的方法，如果不写this也能跑通
        ListNode mid = this.middleNode(head);
        ListNode start2 = this.reverseList(mid);
        ListNode start1 = head;
        while(start2.next != null){
            ListNode nxt1= start1.next;
            ListNode nxt2 = start2.next;
            start1.next = start2;
            start2.next = nxt1;
            start1 = nxt1;
            start2 = nxt2;
        }
        

    }
    //链表的中间节点的方法
    public ListNode middleNode(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
    //链表的反转
    public ListNode reverseList(ListNode head) {
        //方法一
        ListNode pre = null;
        ListNode cur = head;
        while(cur != null){
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }

}
~~~

## 八.前后指针删除链表重复节点

### 237.删除链表中的节点值

因为题目要求，只是删除链表节点的值就行，所以不一定非得移除链表值对应的节点，可以考虑将下一个节点的值copy过来，之后删除下一个节点

~~~java
class Solution {
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
}
~~~

### 19.删除链表的倒数第N个节点：

![image-20240218170946547](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240218170946547.png)

<font color=#00f>**求长度写法，从头遍历链表，获得链表长度，之后确定要删除的节点的位置，再次进行遍历删除节点**</font>

缺点：<font color=#f00>**两次遍历链表，需要判断删除的节点是否是最后一个节点**</font>

~~~java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 * int val;
 * ListNode next;
 * ListNode() {}
 * ListNode(int val) { this.val = val; }
 * ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        // 引入哨兵节点防止删除的节点是第一个节点
        ListNode demmy = new ListNode();
        demmy.next = head;
        ListNode p0 = demmy;
        // 找出链表长度
        ListNode cur = head;
        int nums = 0;
        while (cur != null) {
            nums++;
            cur = cur.next;
        }
        // 找出需要删除的节点的前一个节点
        for (int i = 0; i < nums - n; i++) {
            p0 = p0.next;
        }
        // 我这种想法最大的问题是还需要判断当前是否是结尾
        if (n == 1) {
            p0.next = null;
        } else {
            ListNode nxt = p0.next.next;
            p0.next = nxt;
        }

        return demmy.next;
    }
}
~~~

~~~java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode left = dummy, right = dummy;
        while (n-- > 0)
            right = right.next;
        while (right.next != null) {
            left = left.next;
            right = right.next;
        }
        left.next = left.next.next;
        return dummy.next;
    }
}

~~~

### 83.删除排序链表中的重复元素

~~~java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 * int val;
 * ListNode next;
 * ListNode() {}
 * ListNode(int val) { this.val = val; }
 * ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return head;
        }

        ListNode cur = head;
        while (cur.next != null) {
            if (cur.val == cur.next.val) {
                cur.next = cur.next.next;
            } else {
                cur = cur.next;
            }

        }
        return head;
    }
}
~~~

### 82.删除排序链表中的重复元素II

~~~java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(0, head), cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            int val = cur.next.val;
            if (cur.next.next.val == val)
                while (cur.next != null && cur.next.val == val)
                    cur.next = cur.next.next;
            else
                cur = cur.next;
        }
        return dummy.next;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/solutions/2004067/ru-he-qu-zhong-yi-ge-shi-pin-jiang-tou-p-2ddn/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

~~~java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 * int val;
 * ListNode next;
 * ListNode() {}
 * ListNode(int val) { this.val = val; }
 * ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        // 因为有可能是第一个节点就是重复的需要删除，所以需要引入demmyNode
        ListNode demmy = new ListNode(0, head), cur = demmy;

        // 说明一下cur为什么非得从demmy节点开始，因为必须要保证cur是处于链表不需要删除的部分
        // 需要判断当前cur的值以及下一节点是否存在
        while (cur.next != null && cur.next.next != null) {
            int val = cur.next.val;
            if (cur.next.next.val == val) {
                while (cur.next != null && cur.next.val == val) {
                    cur.next = cur.next.next;
                }
            }else{
                cur = cur.next;
            }
        }
        return demmy.next;
    }
}
~~~

# <font color=#f00 size= 7>二叉树</font>

## 九.二叉树递归本质：

![image-20240218193237128](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240218193237128.png)



### 104.二叉树的最大深度

<font color=#f00>**说明一下空间复杂度与时间复杂度：**</font>

<font color=#f00>**时间复杂度：O(n)因为递归就是将所有的节点的都进行了一次遍历**</font>

<font color=#f00>**空间复杂度：O(n)因为递归过程中是系统在帮忙压栈，所以用到了栈的数据结构，特殊情况下二叉树如果所有节点都只有一个叶子节点那就相当于一个链表了，所以复杂度是O(n)。**</font>

<font color=#00f>**思路一：从下往上递归**</font>，从叶子节点的空节点开始计数为0，这样根节点就是左右节点的深度加1。

~~~java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null){
            return 0;
        }
        return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
    }
}
~~~

<font color=#00f>**思路二：从上往下递归**</font>，递归的同时维护一个全局变量，记录递归到了什么深度，最后递归全部完成就能通过全局变量获得最大深度。

~~~java
class Solution {
    private int ans;

    public int maxDepth(TreeNode root) {
        dfs(root, 0);
        return ans;
    }

    private void dfs(TreeNode node, int cnt) {
        if (node == null) return;
        ++cnt;
        ans = Math.max(ans, cnt);
        dfs(node.left, cnt);
        dfs(node.right, cnt);
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/maximum-depth-of-binary-tree/solutions/2010612/kan-wan-zhe-ge-shi-pin-rang-ni-dui-di-gu-44uz/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

~~~java

class Solution {
    // 维护一个全局变量
    private int res;

    public int maxDepth(TreeNode root) {
        dfs(root, 0);
        return res ;
    }
	// 编写递归逻辑，传递当前节点，以及当前节点所处层数，根节点为1
    private void dfs(TreeNode root, int nums) {
        if (root == null) {
            return;
        }
        nums += 1;
        res = Math.max(res, nums);
        dfs(root.left, nums);
        dfs(root.right,nums);
    }
}
~~~

~~~java
/**
 * 自己写的递归逻辑，有点乱但是一遍就写对了
 */
class Solution {
    private int res;

    public int maxDepth(TreeNode root) {
        dfs(root, 1);
        return res ;
    }

    private void dfs(TreeNode root, int nums) {
        if (root == null) {
            res = Math.max(res, nums-1);
            return;
        }
        dfs(root.left,nums+1);
        dfs(root.right,nums+1);

    }
}
~~~

### 100.<span style="background-color:#00ffff;">相同的树</span>（值得复习，主要看边界条件怎么施加）

![image-20240219200257351](Java重写：灵茶山艾府——基础算法精讲.assets/image-20240219200257351.png)

~~~java
/**
 * 思路：
 * 边界条件左右节点同时为空，回退
 */
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        //这个为什么要这样写边界条件，首先这是一个先序遍历，判断完当前节点后需要判断左右子节点，所以如果左右子节点为空就无法进行判断。
        if (p == null || q == null) {
            return p == q;
        }
        return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);

    }
}
~~~





### ==111.==二叉树的最小深度

![image-20240218202220675](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240218202220675.png)

<font color=#00f>**递推思想**</font>

~~~java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    private int res = Integer.MAX_VALUE;

    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        dfs(root, 1);
        return res;

    }

    private void dfs(TreeNode node, int nums) {

        if (node.left != null) {
            dfs(node.left, nums + 1);
        }
        if (node.right != null) {
            dfs(node.right, nums + 1);
        }
        if(node.right == null && node.left == null){
            res = Math.min(res, nums);
        }
        return;
    }
}
~~~

<font color=#00f>**递归思想**</font>

~~~java
/**
 * 思路：在左右子节点存在的时候进行递归，当前节点的最小深度，
 * 等于左右子节点的最小深度加上自身1个单位的深度
 */
class Solution {
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        int ans = Integer.MAX_VALUE;

        if(root.left != null){
            ans = Math.min(minDepth(root.left),ans);
        }
        if(root.right != null){
            ans = Math.min(minDepth(root.right),ans);
        }
        return ans + 1;
    }
}
~~~

<font color=#00f>**使用队列做的大佬，时间复杂度大大降低**</font>

~~~java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        int res = 0;
        Deque<TreeNode> deque = new LinkedList<>();
        deque.add(root);
        int size = deque.size();
        while (size > 0) {
            res++;
            while (size > 0) {
                root = deque.poll();
                if (root.left == null && root.right == null) {
                    return res;
                }
                if (root.left != null) {
                    deque.add(root.left);
                }
                if (root.right != null) {
                    deque.add(root.right);
                }
                size--;
            }
            size = deque.size();
        }
        return res;
    }
}
~~~

### 112.路径总和I

![image-20240405205521613](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405205521613.png)

~~~java
class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        //先序遍历
        if (root.left == null && root.right == null) {
            return sum == root.val;
        }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }
}

~~~

### ==113.==路径总和II

![image-20240221130500918](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221130500918.png)

<font color=#f00>**这里有一个很重要的知识点，就是在使用路径path的过程中，在向res集合中保存结果时，需要对这个路径进行拷贝，要不然全局变量path会在下次递归的过程中进行改变，如果有回溯操作，那么path最终结果一定是空的，python中也有copy的这一步骤**</font>

~~~java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    int sum = 0;

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        if (root != null) {
            path.add(root.val);
            dfs(root, targetSum);
        }
        return res;
    }

    private void dfs(TreeNode node, int sum) {
        sum -= node.val;
        if (node.left == null && node.right == null) {
            if (sum == 0) {
                res.add(new ArrayList<>(path));
            }
            return;
        }
        if (node.left != null) {
            path.add(node.left.val);
            dfs(node.left, sum);
            path.remove(path.size() - 1);
        }
        if (node.right != null) {
            path.add(node.right.val);
            dfs(node.right, sum);
            path.remove(path.size() - 1);
        }
        return;
    }
}
~~~



### 129.求根节点到叶节点数字之和

![image-20240221110824730](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221110824730.png)

~~~java
class Solution {
    List<Integer> path = new ArrayList<>();
    int res = 0;
    public int sumNumbers(TreeNode root) {
        // 如果节点为0，那么就返回0
        if (root == null) return 0;
        // 首先将根节点放到集合中
        path.add(root.val);
        // 开始递归
        recur(root);
        return res;
    }

    public void recur(TreeNode root){
        if (root.left == null && root.right == null) {
            // 当是叶子节点的时候，开始处理
            res += listToInt(path);
            return;
        }

        if (root.left != null){
            // 注意有回溯
            path.add(root.left.val);
            recur(root.left);
            path.remove(path.size() - 1);
        }
        if (root.right != null){
            // 注意有回溯
            path.add(root.right.val);
            recur(root.right);
            path.remove(path.size() - 1);
        }
        return;
    }
    public int listToInt(List<Integer> path){
        int sum = 0;
        for (Integer num:path){
            // sum * 10 表示进位
            sum = sum * 10 + num;
        }
        return sum;
    }
}

作者：代码随想录
链接：https://leetcode.cn/problems/sum-root-to-leaf-numbers/solutions/464953/129-qiu-gen-dao-xie-zi-jie-dian-shu-zi-zhi-he-di-4/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

<font color=#00f>**自己写的回溯方法**</font>

~~~java
class Solution {
    List<Integer> path = new ArrayList<>();
    int res = 0;

    public int sumNumbers(TreeNode root) {
        //如果节点为0，那么就返回0
        if(root == null) return 0;
        path.add(root.val);
        dfs(root);
        return res;

    }
    //记录当前节点到集合中
    private void dfs(TreeNode node){
        //如果当前节点是叶子节点将当前path集合中的值存到res中并回退
        //向这种属于后序遍历，一般边界条件都是判断当前节点是否是叶子节点
        if(node.left == null && node.right == null){
            res += listToInt(path);
            return;
        }
        if(node.left != null){
            path.add(node.left.val);
            dfs(node.left);
            path.remove(path.size()-1);
        }

        if(node.right != null){
            path.add(node.right.val);
            dfs(node.right);
            path.remove(path.size()-1);
        }
        return;
    }

    private int listToInt(List<Integer> path){
        int sum = 0;
        for(Integer num : path){
            sum = sum * 10 + num;
        }
        return sum;
    }
}
~~~

<font color=#f00>**不用回溯方法**</font>

~~~JAVA
class Solution {
    private int sum = 0;
    public int sumNumbers(TreeNode root) {
        if (root != null) {
            dfs(root, 0);
        }
        return sum;
    }

    private void dfs(TreeNode root, int curSum) {
        curSum = curSum * 10 + root.val;
        if (root.left == null && root.right == null) {
            sum += curSum;
            return;
        }
        if (root.left != null) dfs(root.left, curSum);
        if (root.right != null) dfs(root.right, curSum);
    }
}
~~~

### ==257.==二叉树所有路径

![image-20240221113902519](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221113902519.png)

<font color=#00f>**自己写的递归方法**</font>

~~~java
class Solution {
    List<String> res = new ArrayList<>();

    public List<String> binaryTreePaths(TreeNode root) {
        dfs(root,"");
        return res;
    }
    private void dfs(TreeNode node, String str){
        str += (str.isEmpty() ? "" : "->") + node.val;
        if(node.left == null && node.right == null){
            res.add(str);
            return;
        }
        if(node.left != null){
            dfs(node.left, str);
        }
        if(node.right != null){
            dfs(node.right, str);
        }
        return;
    }
}
~~~

<font color=#f00>**题解使用StringBuffer的方法**</font>

~~~java
class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> paths = new ArrayList<String>();
        constructPaths(root, "", paths);
        return paths;
    }

    public void constructPaths(TreeNode root, String path, List<String> paths) {
        if (root != null) {
            StringBuffer pathSB = new StringBuffer(path);
            pathSB.append(Integer.toString(root.val));
            if (root.left == null && root.right == null) {  // 当前节点是叶子节点
                paths.add(pathSB.toString());  // 把路径加入到答案中
            } else {
                pathSB.append("->");  // 当前节点不是叶子节点，继续递归遍历
                constructPaths(root.left, pathSB.toString(), paths);
                constructPaths(root.right, pathSB.toString(), paths);
            }
        }
    }
}

作者：力扣官方题解
链接：https://leetcode.cn/problems/binary-tree-paths/solutions/400326/er-cha-shu-de-suo-you-lu-jing-by-leetcode-solution/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



### 1448.统计二叉树中好节点的数目（不用复习）

![image-20240221114103214](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221114103214.png)

<font color=#00f>**一遍写过的，使用前序遍历，前序遍历的边界条件一般是node==null时return**</font>

<font color=#f00>**后序遍历的边界条件一般是判断当前节点是否是子节点，这句话有问题不管前中后序遍历边界条件都可以是node==null时return，有的时候因为边界条件可以用判断是否是叶子节点代替，或者说是叶子节点时才进行操作所以有的时候可以更换边界条件**</font>

~~~java
class Solution {
    private int ans = 0;

    public int goodNodes(TreeNode root) {
        dfs(root,Integer.MIN_VALUE);
        return ans;
    }

    private void dfs(TreeNode node, int num) {
        if (node == null) {
            return;
        }
        if (node.val >= num) {
            ans += 1;
        }
        num = Math.max(num, node.val);
        dfs(node.left, num);
        dfs(node.right, num);
    }
}
~~~



### ==987.==二叉树的垂序遍历（困难题）

![image-20240221115555958](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221115555958.png)

<font color=#f00>**很多的高级语法，要多加练习**</font>

~~~java
class Solution {
    public List<List<Integer>> verticalTraversal(TreeNode root) {
        Map<Integer, List<int[]>> groups = new TreeMap<>();
        dfs(root, 0, 0, groups);

        List<List<Integer>> ans = new ArrayList<>(groups.size());
        for (List<int[]> g : groups.values()) {
            // 重写sort排序方法
            g.sort((a, b) -> a[0] != b[0] ? a[0] - b[0] : a[1] - b[1]);
            List<Integer> vals = new ArrayList<>(g.size());
            for (int[] p : g) {
                vals.add(p[1]);
            }
            ans.add(vals);
        }
        return ans;
    }

    private void dfs(TreeNode node, int row, int col, Map<Integer, List<int[]>> groups) {
        if (node == null) {
            return;
        }
        // col 相同的分到同一组
        groups.computeIfAbsent(col, k -> new ArrayList<>()).add(new int[]{row, node.val});
        dfs(node.left, row + 1, col - 1, groups);
        dfs(node.right, row + 1, col + 1, groups);
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/vertical-order-traversal-of-a-binary-tree/solutions/2638913/si-chong-xie-fa-dfsha-xi-biao-shuang-shu-tg6q/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

~~~java
class Solution {
    public List<List<Integer>> verticalTraversal(TreeNode root) {
        Map<Integer, List<int[]>> groups = new TreeMap<>();
        dfs(root, 0, 0, groups);

        List<List<Integer>> ans = new ArrayList<>(groups.size());
        for(List<int[]> g:groups.values()){
            g.sort((a, b) -> a[0] != b[0] ? a[0] - b[0] : a[1] - b[1]);
            List<Integer> vals = new ArrayList<>(g.size());
            for (int[] p : g) {
                vals.add(p[1]);
            }
            ans.add(vals);
        }
        return ans;
    }

    private void dfs(TreeNode node, int row, int col, Map<Integer, List<int[]>> groups) {
        if (node == null) {
            return;
        }
        // col相同的分到同一组
        groups.computeIfAbsent(col, k -> new ArrayList<>()).add(new int[] { row, node.val });
        dfs(node.left, row + 1, col - 1, groups);
        dfs(node.right, row + 1, col + 1, groups);

    }
}
~~~



## ==十.如何灵活运用递归==

### 100.相同的树（不需要复习）

![image-20240418151158175](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418151158175.png)

~~~java
/**
 * 思路：
 * 边界条件左右节点同时为空，回退
 */
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null || q == null) {
            return p == q;
        }
        return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);

    }
}
~~~



### 101.对称二叉树

![image-20240219201210095](Java重写：灵茶山艾府——基础算法精讲.assets/image-20240219201210095.png)

<font color=#f00>**在100题的思路上自己一遍写过的**</font>

~~~java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return this.Symmetric(root.left,root.right);
    }
    private boolean Symmetric(TreeNode left, TreeNode right){
        if(left == null || right == null){
            return left == right;
        }
        return left.val == right.val && Symmetric(left.left, right.right) && Symmetric(left.right,right.left);
    }
}
~~~



### ==110.==平衡二叉树

<font color=#f00>**这道题主要要知道传递的参数，因为要知道左右二叉树不平衡的状态所以可以通过不存在的参数，来表示不满足要求的情况**</font>

![image-20240405212135437](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405212135437.png)

<font color=#00f>**这道题肯定是需要计算左右子树的高度的，所以返回的值肯定是int类型，那如果表示树不平衡，可以利用树的高度是非负，如果返回-1则表示不平衡**</font>

~~~java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return get_height(root) != -1;

    }
    private int get_height(TreeNode node){
        //边界条件
        if(node == null){
            return 0;
        }
        //先判断左子树的高度
        int left_height = get_height(node.left);
        if(left_height == -1){
            return -1;
        }
        int right_height = get_height(node.right);
        if(right_height == -1 || Math.abs(left_height - right_height)> 1){
            return -1;
        } 
        return Math.max(left_height,right_height) + 1;
    }
}
~~~

解释说明：<font color=#00f>**这道题的思路从104.二叉树的最大深度演变过来，对于get_height方法原来是：**</font>

~~~java
class Solution {
    public int get_height(TreeNode root) {
        if(root == null){
            return 0;
        }
        int right_height = get_height(root.right);
        int left_height = get_height(root.left);
        return Math.max(right_height,left_height)+1;
    }
}
~~~

<font color=#00f>**可以看出主要的变化是在计算完左右子树高度后分别加了一个判断，判断左子树相当于剪枝，判断右子树相当于剪枝加判断是否平衡**</font>

<font color=#00f>**这个剪枝可以省略吗：省略的代码如下**</font>

~~~java
/**
省略后的代码只能获得部分正确结果，因为去掉剪枝，当左右其中一颗子树判断不平衡后，相当于一个子树的高度变为-1了，之后用-1剪去另一颗子树的高度永远小于1，不满足判断条件，所以应该当左右任意一颗子树高度为-1时，进行剪枝，剪枝的方式就是在左右子树的高度返回后进行判断是否高度为-1，如果是-1直接return
*/
class Solution {
    public boolean isBalanced(TreeNode root) {
        return get_height(root) != -1;

    }
    private int get_height(TreeNode node){
        //边界条件
        if(node == null){
            return 0;
        }
        //先判断左子树的高度
        int left_height = get_height(node.left);
        int right_height = get_height(node.right);
        if(Math.abs(left_height - right_height)> 1){
            return -1;
        } 
        return Math.max(left_height,right_height) + 1;
    }
}
~~~

<font color=#f00>**剪枝后的代码就是最上面的代码**</font>

<font color=#f00>**按照师兄的思路写的**</font>

~~~JAVA

class Solution {
    private int ans=-1;

    public boolean isBalanced(TreeNode root) {
        ans = 0;
        get_height(root);
        return ans != -1;

    }
    private int get_height(TreeNode node){
        //边界条件
        if(node == null){
            return 0;
        }
        //先判断左子树的高度
        int left_height = get_height(node.left);
        int right_height = get_height(node.right);
        if(Math.abs(left_height - right_height)> 1){
            ans = -1;
        } 
        return Math.max(left_height,right_height) + 1;
    }
}
~~~



### 199.二叉树的右视图

![image-20240220101957860](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240220101957860.png)

![image-20240220102059112](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240220102059112.png)

需要解决问题：

- 从右树开始进行递归
- 怎么把答案记下来
- 怎么判断当前节点是否需要进行记录（记录一个节点深度如果这个节点深度等于答案的长度，就需要进行记录）

~~~java
class Solution {
    private List<Integer> res = new ArrayList<>();

    public List<Integer> rightSideView(TreeNode root) {
        dfs(root, 0);
        return res;

    }

    private void dfs(TreeNode root, int num) {

        if (root == null) {
            return;
        }
        if (num == res.size()) {
            res.add(root.val);
        }
        dfs(root.right, num + 1);
        dfs(root.left, num + 1);
    }
}
~~~

### ==226.==翻转二叉树

![image-20240220112106030](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240220112106030.png)

~~~java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null){
            return null;
        }
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }
}
~~~



### ==1026.==节点与其祖先之间的最大差值

![image-20240222104714046](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240222104714046.png)

![image-20240220115254901](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240220115254901.png)

<font color=#00f>**这种思路是从根节点出发，所以递推的过程中可以确保都是祖先关系**</font>

~~~java
class Solution {
    private int ans;

    public int maxAncestorDiff(TreeNode root) {
        dfs(root, root.val, root.val);
        return ans;
    }

    private void dfs(TreeNode node, int mn, int mx) {
        if (node == null) return;
        // 虽然题目要求「不同节点」，但是相同节点的差值为 0，不会影响最大差值
        // 所以先更新 mn 和 mx，再计算差值也是可以的
        // 在这种情况下，一定满足 mn <= node.val <= mx
        mn = Math.min(mn, node.val);
        mx = Math.max(mx, node.val);
        ans = Math.max(ans, Math.max(node.val - mn, mx - node.val));
        dfs(node.left, mn, mx);
        dfs(node.right, mn, mx);
    }
}
~~~

![image-20240220120221901](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240220120221901.png)

<font color=#f00>**这道题还是用归的思路比较容易想清楚，更明确的是祖先节点**</font>

~~~java
class Solution {
    private int ans;

    public int maxAncestorDiff(TreeNode root) {
        dfs(root);
        return ans;
    }

    private int[] dfs(TreeNode node) {
        if (node == null) // 需要保证空节点不影响 mn 和 mx
            return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE};
        int mn = node.val, mx = mn;
        //dfs返回的是左右节点的最小值和最大值
        var p = dfs(node.left);//var关键字是指让编译器自己推断数据类型，并不是可变数据类型
        var q = dfs(node.right);
        mn = Math.min(mn, Math.min(p[0], q[0]));
        mx = Math.max(mx, Math.max(p[1], q[1]));
        ans = Math.max(ans, Math.max(node.val - mn, mx - node.val));
        return new int[]{mn, mx};
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/maximum-difference-between-node-and-ancestor/solutions/2232367/liang-chong-fang-fa-zi-ding-xiang-xia-zi-wj9v/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



### ==1080.==根到叶路径上的不足节点

![image-20240220133428008](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240220133428008.png)

![image-20240418174800067](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418174800067.png)

<font color=#f00>**注意这种写法有一种很巧妙的判断当前节点是不是叶子节点的方法**</font>

~~~java
class Solution {
    //判断当前节点是否是不足节点
    public TreeNode sufficientSubset(TreeNode root, int limit) {
        limit -= root.val;
        if (root.left == root.right) // root 是叶子，root.left和root.right获得都是节点对象，如果不是null是不可能相等的，会进行地址的比较。
            // 如果 limit > 0 说明从根到叶子的路径和小于 limit，删除叶子，否则不删除
            return limit > 0 ? null : root;
        //首先判断左右节点是否是不足节点，只有将当前节点的子树中的所有节点都进行删除后，才能判断当前节点是否是不足节点
        if (root.left != null) 
            root.left = sufficientSubset(root.left, limit);
        if (root.right != null) 
            root.right = sufficientSubset(root.right, limit);
        // 如果儿子都被删除，就删 root，否则不删 root
        return root.left == null && root.right == null ? null : root;
    }
}

~~~

<font color=#00f>**按照上面的思路自己写的**</font>

~~~java
class Solution {
    public TreeNode sufficientSubset(TreeNode root, int limit) {
        limit -= root.val;
        //边界条件是到达叶子节点，且此条叶子节点的路径和小于limit，则删除此叶子节点
        if (root.left == root.right) {
            return limit > 0 ? null : root;
        }
        //典型后序dfs遍历，先进行左右节点递归，之后对于当前节点进行操作
        if(root.left != null){
            root.left = sufficientSubset(root.left,limit);
        }
        if(root.right != null){
            root.right = sufficientSubset(root.right,limit);
        }
        //对于左右节点递归后，判断当前节点，如果当前节点的左右节点都是null，
        //那证明当前节点变为了新的叶子节点
        //解释一下为什么是新变成的叶子节点，如果不是因为左右子节点被删除则会在边界条件中进行判断
        //在这判断的证明都不是原来的叶子节点
        return root.left == null && root.right == null ? null : root;
    }
}
~~~

<font color=#f00>**题解说明二：主要看如何考虑使用先序遍历还是中序遍历还是后序遍历**</font>

![image-20240220141408338](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240220141408338.png)

~~~java
public class Solution2 {

    /**
     * @param node
     * @param s
     * @param limit
     * @return 返回 node 结点是否被删除（注意：这个返回值的意义，直接影响整个逻辑。）
     */
    private Boolean dfs(TreeNode node, int s, int limit) {
        if (node.left == null && node.right == null) {
            return s + node.val < limit;
        }

        // 注意：如果 dfs 的返回值的意义是这个结点是否被删除，它们的默认值应该设置为 true
        boolean lTreeDeleted = true;
        boolean rTreeDeleted = true;

        // 如果有左子树，就先递归处理左子树
        if (node.left != null) {
            lTreeDeleted = dfs(node.left, s + node.val, limit);
        }
        // 如果有右子树，就先递归处理右子树
        if (node.right != null) {
            rTreeDeleted = dfs(node.right, s + node.val, limit);
        }

        // 左右子树是否保留的结论得到了，由自己来执行是否删除它们
        if (lTreeDeleted) {
            node.left = null;
        }
        if (rTreeDeleted) {
            node.right = null;
        }

        // 只有左右子树都被删除了，自己才没有必要保留
        return lTreeDeleted && rTreeDeleted;
    }

    public TreeNode sufficientSubset(TreeNode root, int limit) {
        boolean rootDeleted = dfs(root, 0, limit);
        if (rootDeleted) {
            return null;
        }
        return root;
    }
}

作者：liweiwei1419
链接：https://leetcode.cn/problems/insufficient-nodes-in-root-to-leaf-paths/solutions/8726/hou-xu-bian-li-python-dai-ma-java-dai-ma-by-liweiw/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



### ==1110.==删点成林

![image-20240220135345278](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240220135345278.png)

~~~java
class Solution {
    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        List<TreeNode> ans = new ArrayList<TreeNode>();
        Set<Integer> s = new HashSet<Integer>();
        //将数组中的元素都放在hashset中
        for(int x: to_delete){
            s.add(x);
        }
        if(dfs(ans, s, root) != null) ans.add(root);
        return ans;
    }
    //这道题有个关键的点，就是要想清楚为什么要用归的方式：
    //如果用递的方式，会将一部分未进行判断的子树直接放入到ans中，所以只能用归，而且要先判断左右子树是否是需要删除的
    //dfs含义判断当前节点是应该添加到ans集合中还是应该删除
    //这里把ans，和set集合放在参数中也就是为了维护全局变量，可以直接将其提出到函数外部
    private TreeNode dfs(List<TreeNode> ans, Set<Integer> s, TreeNode node){
        if(node == null){
            return null;
        }
        //后序遍历，先递归左右儿子，
        node.left = dfs(ans, s, node.left);
        node.right = dfs(ans, s, node.right);
        //对于当前节点，如果这个节点不在删除的部分，就直接返回
        if(!s.contains(node.val)) {return node;}
        //运行到这证明这个节点在删除的部分，如果这个节点的左右儿子有，就将这个节点的左右儿子添加到ans中
        if(node.left != null){ans.add(node.left);}
        if(node.right != null){ans.add(node.right);}
        return null;
    }
}

~~~

<font color=#00f>**自己写的方法**</font>

~~~java
class Solution {
    List<TreeNode> ans = new ArrayList<>();
    Set<Integer> s = new HashSet<Integer>();

    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        // 将数组中的元素都放在hashset中
        for (int x : to_delete) {
            s.add(x);
        }
        if (dfs(root) != null)
            ans.add(root);
        return ans;
    }

    // dfs含义判断当前节点是应该添加到ans集合中还是应该删除
    private TreeNode dfs(TreeNode node) {
        if (node == null) {
            return null;
        }
        // 后序遍历，先递归左右儿子，
        node.left = dfs(node.left);
        node.right = dfs(node.right);
        // 对于当前节点，如果这个节点不在删除的部分，就直接返回
        if (!s.contains(node.val)) {
            return node;
        }
        // 运行到这证明这个节点在删除的部分，如果这个节点的左右儿子有，就将这个节点的左右儿子添加到ans中
        if (node.left != null) {
            ans.add(node.left);
        }
        if (node.right != null) {
            ans.add(node.right);
        }
        return null;
    }
}
~~~



### ==1372.==二叉树中的最长交错路径

![image-20240220135520586](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240220135520586.png)

<font color=#00f>**维护一个方向参数，一个到当前节点走过的长度参数**</font>

~~~java
class Solution {
    int max = 0;
    public int longestZigZag(TreeNode root) {
        if (root == null) {
            return 0;
        }
        dfs(root, true, 0);
        return max;
    }

    public void dfs(TreeNode root, boolean isLeft, int sum) {
        if (root == null) {
            return;
        }
        max = Math.max(max, sum);
        
        // 若本级节点是由上级的左子树来的，那么，它的左子树从1开始，右子树从sum+1开始
        if (isLeft) {
            dfs(root.left, true, 1);
            dfs(root.right, false, sum + 1);
        } else {
            dfs(root.left, true, sum + 1);
            dfs(root.right, false, 1);
        }
    }
}
~~~

### ==2385.==感染二叉树需要的总时间

![image-20240429153420351](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240429153420351.png)

<font color=#f00>**做法一，两次遍历**</font>

~~~java
class Solution {
    private TreeNode startNode;
    private final Map<TreeNode, TreeNode> fa = new HashMap<>();

    public int amountOfTime(TreeNode root, int start) {
        dfs(root, null, start);
        return maxDepth(startNode, startNode);
    }
	// 第一遍递归建立邻接表，因为这道题有树的结构所以，就记录当前节点的父节点,还得找到开始节点
    private void dfs(TreeNode node, TreeNode from, int start) {
        if (node == null) {
            return;
        }
        fa.put(node, from); // 记录每个节点的父节点
        if (node.val == start) {
            startNode = node; // 找到 start
        }
        dfs(node.left, node, start);
        dfs(node.right, node, start);
    }
	//第二遍按照图的方式遍历，找到最大的深度
    private int maxDepth(TreeNode node, TreeNode from) {
        if (node == null) {
            return -1; // 注意这里是 -1，因为 start 的深度为 0
        }
        int res = -1;
        if (node.left != from) {
            res = Math.max(res, maxDepth(node.left, node));
        }
        if (node.right != from) {
            res = Math.max(res, maxDepth(node.right, node));
        }
        if (fa.get(node) != from) {
            res = Math.max(res, maxDepth(fa.get(node), node));
        }
        return res + 1;
    }
}

//24/4/29每日一题，建议反复复习
~~~

<font color=#f00>**做法二，一次遍历**</font>

~~~java
class Solution {
    private int ans;

    public int amountOfTime(TreeNode root, int start) {
        dfs(root, start);
        return ans;
    }

    private int[] dfs(TreeNode node, int start) {
        if (node == null) {
            return new int[]{0, 0};
        }
        int[] leftRes = dfs(node.left, start);
        int[] rightRes = dfs(node.right, start);
        int lLen = leftRes[0], lFound = leftRes[1];
        int rLen = rightRes[0], rFound = rightRes[1];
        if (node.val == start) {
            // 计算子树 start 的最大深度
            // 注意这里和方法一的区别，max 后面没有 +1，所以算出的也是最大深度
            ans = Math.max(lLen, rLen);
            return new int[]{1, 1}; // 找到了 start
        }
        if (lFound == 1 || rFound == 1) {
            // 只有在左子树或右子树包含 start 时，才能更新答案
            ans = Math.max(ans, lLen + rLen); // 两条链拼成直径
            // 保证 start 是直径端点
            return new int[]{(lFound == 1 ? lLen : rLen) + 1, 1};
        }
        return new int[]{Math.max(lLen, rLen) + 1, 0};
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/amount-of-time-for-binary-tree-to-be-infected/solutions/2753470/cong-liang-ci-bian-li-dao-yi-ci-bian-li-tmt0x/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



## 十一.二叉搜索树

### ==98.==验证二叉搜素树

<font color=#f00>**可以多写几遍搞懂后序遍历**</font>

<font color=#00f>**前序遍历**</font>

~~~java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean isValidBST(TreeNode node, long left, long right) {
        if (node == null)
            return true;
        long x = node.val;
        return left < x && x < right &&
                isValidBST(node.left, left, x) &&
                isValidBST(node.right, x, right);
    }
}
~~~

<font color=#00f>**中序遍历：对于二叉搜素树来说，中序遍历会得到一个严格递增的数组**</font>

![image-20240221134352226](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221134352226.png)

~~~java
class Solution {
    private long pre = Long.MIN_VALUE;

    //判断当前节点及其子树是不是二叉搜索树
    public boolean isValidBST(TreeNode root) {
        //边界条件
        if(root == null){
            return true;
        }
        //1.先进行左子节点递归
        //2.对于当前节点进行判断，本来是应该写两步的，这个地方进行合并了
        if(!isValidBST(root.left) || root.val <= pre){
            return false;
        }
        pre = root.val;
        //3.对于右子节点进行判断
        return isValidBST(root.right);
    }
}
~~~

<font color=#00f>**后序遍历**</font>

~~~java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return dfs(root)[1] != Long.MAX_VALUE;
    }

    private long[] dfs(TreeNode node) {
        if (node == null)
            return new long[]{Long.MAX_VALUE, Long.MIN_VALUE};
        long[] left = dfs(node.left);
        long[] right = dfs(node.right);
        long x = node.val;
        // 也可以在递归完左子树之后立刻判断，如果发现不是二叉搜索树，就不用递归右子树了
        if (x <= left[1] || x >= right[0])
            return new long[]{Long.MIN_VALUE, Long.MAX_VALUE};
        return new long[]{Math.min(left[0], x), Math.max(right[1], x)};
    }
}

~~~

![image-20240221135611394](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221135611394.png)

### 230.二叉搜索树中第K小的元素

![image-20240221135722816](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221135722816.png)

<font color=#00f>**思路一：中序遍历直接放在集合中（此时集合严格递增），然后取第k个元素**</font>

~~~java
class Solution {
    List<Integer> res = new ArrayList<>();

    public int kthSmallest(TreeNode root, int k) {
        //System.out.println(res);
        res.clear();
        dfs(root);
        return res.get(k-1);
    }

    private void dfs(TreeNode node){
        if(node == null){
            return;
        }
        dfs(node.left);
        res.add(node.val);
        dfs(node.right);
        return;
    }
}
~~~

<font color=#f00>**思路二：用栈做，非递归**</font>

~~~java
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        //中序
        Deque<TreeNode> stack=new LinkedList<>();
        while(root!=null||!stack.isEmpty()){
            while(root!=null){
                stack.push(root);
                root=root.left;
            }
            root=stack.pop();
            k-=1;
            if(k==0)
                break;
            root=root.right;
        }
        return root.val;
    }

}
~~~



### ==501.==二叉搜素树中的众数（==学习map的基础操作语法==）

<font color=#00f>**挺难的简单题**</font>

![image-20240221144016733](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221144016733.png)

~~~java
class Solution {
    Map<Integer, Integer> res = new HashMap<>();

    public int[] findMode(TreeNode root) {
        res.clear();
        dfs(root);
        int maxValue = Integer.MIN_VALUE;
        List<Integer> maxKeys = new ArrayList<>();
        
        for (Map.Entry<Integer, Integer> entry : res.entrySet()) {
            if (entry.getValue() > maxValue) {
                maxKeys.clear();
                maxKeys.add(entry.getKey());
                maxValue = entry.getValue();
            } else if (entry.getValue().equals(maxValue)) {
                maxKeys.add(entry.getKey());
            }
        }

        return maxKeys.stream().mapToInt(Integer::intValue).toArray();
    }
	//中序遍历获得一个单调递增的map
    private void dfs(TreeNode node) {
        if (node == null) {
            return;
        }
        dfs(node.left);
        //仔细学学对于map的compute方法
        res.compute(node.val, (key, count) -> count == null ? 1 : count + 1);
        dfs(node.right);
    }
}
~~~



### 530.二叉搜索树的最小绝对差

![image-20240418181740326](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418181740326.png)

~~~java
// 递归
class Solution {
    TreeNode pre;// 记录上一个遍历的结点
    int result = Integer.MAX_VALUE;
    public int getMinimumDifference(TreeNode root) {
       if(root==null)return 0;
       traversal(root);
       return result;
    }
    public void traversal(TreeNode root){
        if(root==null)return;
        //左
        traversal(root.left);
        //中，中序遍历是单调的所以就是相邻的遍历节点之间的差是最小的
        if(pre!=null){
            result = Math.min(result,root.val-pre.val);
        }
        pre = root;
        //右
        traversal(root.right);
    }
}
~~~



### 700.二叉搜索树中的搜索

![image-20240222105737702](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240222105737702.png)

~~~java
class Solution {
    TreeNode rootans = new TreeNode();
    boolean b = false;

    public TreeNode searchBST(TreeNode root, int val) {
        dfs(root, val);
        if (b == false)
            return null;
        return rootans;
    }

    private void dfs(TreeNode node, int val) {
        if (node == null) {
            return;
        }
        if (node.val == val) {
            rootans = node;
            b = true;
            return;
        }
        dfs(node.left, val);
        dfs(node.right, val);
    }

}
~~~

### ==1373.==二叉搜索子树的最大键值和

![image-20240222105949020](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240222105949020.png)

~~~java
class Solution {
    private int ans; // 二叉搜索树可以为空

    public int maxSumBST(TreeNode root) {
        dfs(root);
        return ans;
    }

    private int[] dfs(TreeNode node) {
        if (node == null)
            return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE, 0};

        int[] left = dfs(node.left); // 递归左子树
        int[] right = dfs(node.right); // 递归右子树
        int x = node.val;
        if (x <= left[1] || x >= right[0]) // 不是二叉搜索树
            return new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE, 0};

        int s = left[2] + right[2] + x; // 这棵子树的所有节点值之和
        ans = Math.max(ans, s);

        return new int[]{Math.min(left[0], x), Math.max(right[1], x), s};
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/maximum-sum-bst-in-binary-tree/solutions/2276783/hou-xu-bian-li-pythonjavacgo-by-endlessc-gll3/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~





## 十二.二叉树公共祖先问题

### ==236.==二叉树的最近公共祖先

![image-20240222150912772](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240222150912772.png)

![image-20240222134226781](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240222134226781.png)

~~~java
class Solution {
    //递归函数的含义当前节点及其子树是否有p,q两节点，如果有就返回当前节点，如果没有就返回null
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left != null && right != null){
            return root;
        }
        if(left != null){
            return left;
        }
        if(right != null){
            return right;
        }
        return null;
    }
}
~~~



### 235.二叉搜索树的最近公共祖先

![image-20240222150927117](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240222150927117.png)

~~~java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        int x = root.val;
        if (p.val < x && q.val < x) {
            return lowestCommonAncestor(root.left, p, q);
        }
        if (p.val > x && q.val > x) {
            return lowestCommonAncestor(root.right, p, q);
        }
        return root;
    }
}
~~~



![image-20240221135551598](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221135551598.png)

### ==865/1123.==最深叶节点的最近公共祖先

![image-20240223072052748](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240223072052748.png)

![image-20240418205456917](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418205456917.png)

<font color=#00f>**维护两个参数一个参数记录当前节点所在的深度，一个记录左右子树各自的深度**</font>

~~~java
class Solution {
    private TreeNode ans;
    private int maxDepth = -1; // 全局最大深度

    public TreeNode subtreeWithAllDeepest(TreeNode root) {
        dfs(root, 0);
        return ans;
    }

    private int dfs(TreeNode node, int depth) {
        if (node == null) {
            maxDepth = Math.max(maxDepth, depth); // 维护全局最大深度
            return depth;
        }
        int leftMaxDepth = dfs(node.left, depth + 1); // 获取左子树最深叶节点的深度
        int rightMaxDepth = dfs(node.right, depth + 1); // 获取右子树最深叶节点的深度
        if (leftMaxDepth == rightMaxDepth && leftMaxDepth == maxDepth)
            ans = node;
        return Math.max(leftMaxDepth, rightMaxDepth); // 当前子树最深叶节点的深度
    }
}

~~~



### ==2096.==从二叉树一个节点到另一个节点每一步的方向（挺不错的题）

![image-20240418194754445](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418194754445.png)

~~~java
class Solution {
    public String getDirections(TreeNode root, int startValue, int destValue) {
//        我的思路是肯定是自上而下递归的，startValue是相反的，那我可以直接传相反，
        ansOfDir = new StringBuffer();
        root=getTargetNode(root, startValue, destValue);
        path = new ArrayList<>();
        getStartValue(root,startValue,0);
        getDestValue(root,destValue);

        return ansOfDir.toString();
    }
    StringBuffer ansOfDir;

    List<String> path;
    public void getStartValue(TreeNode node,int target,int high)
    {
        if (node==null)
        {
            return;
        }
        if (node.val==target)
        {
            while (high>0)
            {
                ansOfDir.append("U");
                high--;
            }
            return;
        }
        getStartValue(node.left , target,high+1);
        getStartValue(node.right, target,high+1);
    }

    public void getDestValue(TreeNode node,int target)
    {
        if (node==null)
        {
            return;
        }
        if (node.val==target)
        {
            for (String s :path) {
                ansOfDir.append(s);
            }

            return;
        }
        path.add("L");
        getDestValue(node.left, target);
        path.remove(path.size()-1);
        path.add("R");
        getDestValue(node.right, target);
        path.remove(path.size()-1);
    }
    private TreeNode getTargetNode(TreeNode node,int tar1,int tar2)
    {
        if (node==null||node.val==tar1||node.val==tar2)
        {
            return node;
        }
        TreeNode l  = getTargetNode(node.left, tar1, tar2);
        TreeNode r  = getTargetNode(node.right, tar1, tar2);
        if (l!=null&&r!=null)
        {
            return node;
        }
        return l!=null?l:r;
    }
}
~~~



## 十三.二叉树层序遍历

![image-20240221110421693](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221110421693.png)

### ==102.==二叉树的层序遍历

![image-20240222201231733](Java重写：灵茶山艾府——基础算法精讲.assets/image-20240222201231733.png)

~~~java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        if(root == null) return List.of();//这样返回的是空集合
        List<List<Integer>> ans = new ArrayList<>();
        List<TreeNode> cur = List.of(root);//将根节点放入集合
        while(!cur.isEmpty()){
            List<TreeNode> nxt = new ArrayList<>();
            List<Integer> vals = new ArrayList<>(cur.size());//预分配空间
            for(TreeNode node : cur){
                vals.add(node.val);
                if(node.left != null) nxt.add(node.left);
                if(node.right != null) nxt.add(node.right);
            }
            cur = nxt;
            ans.add(vals);
        }
        return ans;
    }
}
~~~

<font color=#f00>**使用队列的方法**</font>

~~~java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        if(root == null) return List.of();//这样返回的是空集合
        List<List<Integer>> ans = new ArrayList<>();
        Queue<TreeNode> q = new ArrayDeque<>();
        q.add(root);//将根节点放入集合
        while(!q.isEmpty()){
            int n = q.size();
            List<Integer> vals = new ArrayList<>(n);//预分配空间
            while(n-- > 0){
                TreeNode node = q.poll();
                vals.add(node.val);
                if(node.left != null) q.add(node.left);
                if(node.right != null) q.add(node.right);
            }
            ans.add(vals);
        }
        return ans;
    }
}
~~~



### 103.二叉树的锯齿形层序遍历

![image-20240418211654767](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418211654767.png)

~~~java
class Solution {
    //规定一个boolean值，true往右，false往左
    boolean dir = false;
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if(root == null) return List.of();
        List<List<Integer>> ans = new ArrayList<>();
        List<TreeNode> cur = List.of(root);
        while(!cur.isEmpty()){
            List<TreeNode> nxt = new ArrayList<>();
            List<Integer> vals = new ArrayList<>(cur.size());
            for(TreeNode node : cur){
                vals.add(node.val);
                if(node.left != null) nxt.add(node.left);
                if(node.right != null) nxt.add(node.right);
            }
            cur = nxt;
            if(dir == true) Collections.reverse(vals);
            ans.add(vals);
            dir = !dir;
        }
        return ans;
    }
}
~~~

~~~java
class Solution {
    //规定一个boolean值，true往右，false往左
    boolean dir = false;
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if(root == null) return List.of();//这样返回的是空集合
        List<List<Integer>> ans = new ArrayList<>();
        Queue<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        while(!q.isEmpty()){
            int n = q.size();
            List<Integer> vals = new ArrayList<>(n);//预分配空间
            while(n-- > 0){
                TreeNode node = q.poll();
                vals.add(node.val);
                if(node.left != null) q.add(node.left);
                if(node.right != null) q.add(node.right);
            }
            if(dir == true) Collections.reverse(vals);
            ans.add(vals);
            dir = !dir;
        }
        return ans;
    }
}
~~~



### ==513.==找树左下角的值

![image-20240418212401519](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418212401519.png)

<font color=#00f>**笨方法，主要看第二种**</font>

~~~java
class Solution {
    public int findBottomLeftValue(TreeNode root) {
        
        List<List<Integer>> ans = new ArrayList<>();
        List<TreeNode> cur = List.of(root);//将根节点放入集合
        int ans1;
        while(true){
            List<TreeNode> nxt = new ArrayList<>();
            List<Integer> vals = new ArrayList<>(cur.size());//预分配空间
            for(TreeNode node : cur){
                vals.add(node.val);
                if(node.left != null) nxt.add(node.left);
                if(node.right != null) nxt.add(node.right);
            }
            if(nxt.isEmpty()){
                ans1 = vals.get(0);
                break;
            }
            cur = nxt;
            ans.add(vals);
        }
        return ans1;
    }
}
~~~

~~~java
class Solution {
    public int findBottomLeftValue(TreeNode root) {
        TreeNode node = root;
        Queue<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        while (!q.isEmpty()) {
            node = q.poll();
            //注意只要先放右节点再放左节点，这样找最后一个节点就是最左侧的节点
            if (node.right != null) q.add(node.right);
            if (node.left != null)  q.add(node.left);
        }
        return node.val;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/find-bottom-left-tree-value/solutions/2049776/bfs-wei-shi-yao-yao-yong-dui-lie-yi-ge-s-f34y/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



![image-20240221135523104](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221135523104.png)

### ==107.==二叉树的层序遍历II

![image-20240223080655095](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240223080655095.png)

<font color=#00f>**最简单的方法层序遍历之后翻转集合**</font>

~~~java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if(root == null) return List.of();//这样返回的是空集合
        List<List<Integer>> ans = new ArrayList<>();
        Queue<TreeNode> q = new ArrayDeque<>();
        q.add(root);//将根节点放入集合
        while(!q.isEmpty()){
            int n = q.size();
            List<Integer> vals = new ArrayList<>(n);//预分配空间
            while(n-- > 0){
                TreeNode node = q.poll();
                vals.add(node.val);
                if(node.left != null) q.add(node.left);
                if(node.right != null) q.add(node.right);
            }
            ans.add(vals);
        }
        Collections.reverse(ans);
        return ans;
    }
}
~~~

<font color=#00f>**使用dfs**</font>

~~~java
class Solution {
    private List<List<Integer>> ans = new ArrayList<>();
    //使用dfs方法
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        dfs(root,0);
        Collections.reverse(ans);
        return ans;
    }
    private void dfs(TreeNode node , int level){
        if(node == null){
            return;
        }
        //当遍历到比ans集合长的层时，创建一个新的层
        if(ans.size() == level) ans.add(new ArrayList<>());
        //将遍历到的节点根据level层数，添加到对应的ans子集合中去
        ans.get(level).add(node.val);
        //继续遍历其他节点
        dfs(node.left, level + 1);
        dfs(node.right, level + 1);
        return;
    }
}
~~~

![image-20240221135523104](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240221135523104.png)

### ==116.==填充每个节点的下一个右侧节点指针

![image-20240223083141637](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240223083141637.png)

<font color=#00f>**dfs方法：值得一看**</font>

~~~java
class Solution {
    // 这个集合用来存储，集合的第几个元素代表的存储树的第几层，
    // 这一层的元素都会存储在这一个位置，随着递归的进行，这个位置存储的节点是不断往右更新的
    private final List<Node> pre = new ArrayList<>();

    public Node connect(Node root) {
        dfs(root,0);
        return root;
    }

    private void dfs(Node node, int depth) {
        if (node == null) {
            return;
        }
        if (depth == pre.size()) {
            pre.add(node);
        } else {
            pre.get(depth).next = node;
            pre.set(depth, node);
        }

        dfs(node.left, depth + 1);
        dfs(node.right, depth + 1);

    }
}
~~~



<font color=#00f>**bfs方法**</font>

~~~java
class Solution {
    public Node connect(Node root) {
        if(root == null){
            return null;
        }
        List<Node> cur = List.of(root);
        while(!cur.isEmpty()){
            List<Node> nxt = new ArrayList<>();
            for(int i = 0; i < cur.size(); i++){
                Node temp = cur.get(i);
                //连接同一层的相邻的两个节点
                if(i>0){
                    cur.get(i-1).next = temp;
                }

                if(temp.left != null){
                    nxt.add(temp.left);
                }
                if(temp.right != null){
                    nxt.add(temp.right);
                }
            }
            cur = nxt;
        }
        return root;
    }
}
~~~



### 117.填充每个节点的下一个右侧节点指针II

<font color=#00f>**dfs方法同上一题，一遍过**</font>

~~~java

class Solution {
    //dfs方法同116完全一样
    private final List<Node> pre = new ArrayList<>();

    public Node connect(Node root) {
        dfs(root,0);
        return root;
    }
    private void dfs(Node node, int depth){
        if(node == null){
            return;
        }
        if(depth == pre.size()){
            pre.add(node);
        }else{
            pre.get(depth).next = node;
            pre.set(depth, node);
        }
        dfs(node.left, depth + 1);
        dfs(node.right,depth + 1);
    }
}
~~~

<font color=#00f>**bfs方法**</font>

~~~java
class Solution {
    public Node connect(Node root) {
        if(root == null) return null;
        List<Node> cur = List.of(root);
        while(!cur.isEmpty()){
            List<Node> nxt = new ArrayList<>();
            for(int i=0; i < cur.size(); i++){
                Node node = cur.get(i);
                if(i>0){
                    cur.get(i-1).next = node;
                }
                if (node.left != null) {
                    nxt.add(node.left);
                }
                if (node.right != null) {
                    nxt.add(node.right);
                }
            }
            cur = nxt;
        }
        return root;
    }
}
~~~



### 1302.层数最深的叶子节点的和

![image-20240223090814077](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240223090814077.png)

<font color=#00f>**bfs方法**</font>

~~~java
class Solution {
    public int deepestLeavesSum(TreeNode root) {
        Map<Integer, Integer> map = new HashMap<>();
        Deque<TreeNode> d = new ArrayDeque<>();
        d.addLast(root);
        int depth = 0;
        while (!d.isEmpty()) {
            int sz = d.size();
            while (sz-- > 0) {
                TreeNode node = d.pollFirst();
                map.put(depth, map.getOrDefault(depth, 0) + node.val);
                if (node.left != null) d.addLast(node.left);
                if (node.right != null) d.addLast(node.right);
            }
            depth++;
        }
        return map.get(depth - 1);
    }
}

作者：宫水三叶
链接：https://leetcode.cn/problems/deepest-leaves-sum/solutions/1754227/by-ac_oier-srst/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

<font color=#00f>**dfs方法**</font>

~~~java
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    int max;
    public int deepestLeavesSum(TreeNode root) {
        dfs(root, 0);
        return map.get(max);
    }
    void dfs(TreeNode root, int depth) {
        if (root == null) return ;
        max = Math.max(max, depth);
        map.put(depth, map.getOrDefault(depth, 0) + root.val);
        dfs(root.left, depth + 1);
        dfs(root.right, depth + 1);
    }
}

作者：宫水三叶
链接：https://leetcode.cn/problems/deepest-leaves-sum/solutions/1754227/by-ac_oier-srst/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

~~~java
class Solution {
    int result=0,maxLevel=0;
    public int deepestLeavesSum(TreeNode root) {
        calculate(root,maxLevel);
        return result;
    }
    public void calculate(TreeNode root,int level){
        if(level>maxLevel){
            maxLevel=level;
            result=root.val;
        }else if(level==maxLevel){
            result+=root.val;
        }
        if(root.left!=null) calculate(root.left,level+1);
        if(root.right!=null) calculate(root.right,level+1);
    }
}
~~~

### 1609.奇偶树

![image-20240223103856558](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240223103856558.png)

~~~java
class Solution {
    public boolean isEvenOddTree(TreeNode root) {
        Deque<TreeNode> d = new ArrayDeque<>();
        boolean flag = true;
        d.addLast(root);
        while (!d.isEmpty()) {
            int size = d.size(), prev = flag ? 0 : 0x3f3f3f3f;
            while (size-- > 0) {
                TreeNode node = d.pollFirst();
                int cur = node.val;
                if (flag && (cur % 2 == 0 || cur <= prev)) return false;
                if (!flag && (cur % 2 != 0 || cur >= prev)) return false;
                prev = cur;
                if (node.left != null) d.addLast(node.left);
                if (node.right != null) d.addLast(node.right);
            }
            flag = !flag;
        }
        return true;
    }
}

~~~

<font color=#00f>**dfs解法**</font>

~~~java
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    public boolean isEvenOddTree(TreeNode root) {
        return dfs(root, 0);
    }
    boolean dfs(TreeNode root, int idx) {
        boolean flag = idx % 2 == 0;
        int prev = map.getOrDefault(idx, flag ? 0 : 0x3f3f3f3f), cur = root.val;
        if (flag && (cur % 2 == 0 || cur <= prev)) return false;
        if (!flag && (cur % 2 != 0 || cur >= prev)) return false;
        map.put(idx, root.val);
        if (root.left != null && !dfs(root.left, idx + 1)) return false;
        if (root.right != null && !dfs(root.right, idx + 1)) return false;
        return true;
    }
}

~~~



### 2415.反转二叉树的奇数层

![image-20240223124525921](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240223124525921.png)

![image-20240223133523339](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240223133523339.png)

~~~java
List<Integer> q = List.of(root);
int level = 0;
class Solution {
    public TreeNode reverseOddLevels(TreeNode root) {
        List<TreeNode> cur = List.of(root);
        int level = 0;
        while (!cur.isEmpty()) {
            List<TreeNode> nxt = new ArrayList<>();
            for (TreeNode node : cur) {
                if (node.left != null) {
                    nxt.add(node.left);
                    nxt.add(node.right);
                }
            }
            cur = nxt;
            if (level == 0) {
                for (int i = 0; i < cur.size() / 2; i++) {
                    TreeNode x = cur.get(i), y = cur.get(cur.size() - 1 - i);
                    int temp = 0;
                    temp = x.val;
                    x.val = y.val;
                    y.val = temp;
                }
            }
            //注意按位异或运算是在二进制下，对应位相同返回0，对应位不同该位变为1；
            level ^= 1;//这个作用是让level在0,1之间反复横跳
        }
        return root;
    }
}
~~~

<font color=#f00>**dfs**</font>

~~~java
class Solution:
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(node1: Optional[TreeNode], node2: Optional[TreeNode], is_odd_level: bool) -> None:
            if node1 is None: return
            if is_odd_level: node1.val, node2.val = node2.val, node1.val
            dfs(node1.left, node2.right, not is_odd_level)
            dfs(node1.right, node2.left, not is_odd_level)
        dfs(root.left, root.right, True)
        return root

作者：灵茶山艾府
链接：https://leetcode.cn/problems/reverse-odd-levels-of-binary-tree/solutions/1831556/zhi-jie-jiao-huan-zhi-by-endlesscheng-o8ze/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



### 2641.二叉树的堂兄弟节点II

![image-20240418215610000](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418215610000.png)

~~~java
class Solution {
    public TreeNode replaceValueInTree(TreeNode root) {
        root.val = 0;
        List<TreeNode> q = List.of(root);
        while (!q.isEmpty()) {
            List<TreeNode> tmp = q;
            q = new ArrayList<>();

            // 计算下一层的节点值之和
            int nextLevelSum = 0;
            for (TreeNode node : tmp) {
                if (node.left != null) {
                    q.add(node.left);
                    nextLevelSum += node.left.val;
                }
                if (node.right != null) {
                    q.add(node.right);
                    nextLevelSum += node.right.val;
                }
            }

            // 再次遍历，更新下一层的节点值
            for (TreeNode node : tmp) {
                int childrenSum = (node.left != null ? node.left.val : 0) +
                                  (node.right != null ? node.right.val : 0);
                if (node.left != null) node.left.val = nextLevelSum - childrenSum;
                if (node.right != null) node.right.val = nextLevelSum - childrenSum;
            }
        }
        return root;
    }
}

~~~



# <font color=#f00 size= 7>回溯</font>

## 十四.子集型回溯

回溯有一个<font color=#f00>**增量构造答案**</font>的过程，是用递归实现。

递归：只要把边界条件写对，和递归的逻辑写对，就一定能得到答案

### 17.电话号码的字母组合

![image-20240225103549270](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240225103549270.png)

~~~java
class Solution {
    private static final String[] MAPPING = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

    private final List<String> ans = new ArrayList<>();
    private char[] digits, path;

    public List<String> letterCombinations(String digits) {
        int n = digits.length();
        if (n == 0) return List.of();
        this.digits = digits.toCharArray();
        path = new char[n]; // 本题 path 长度固定为 n
        dfs(0);
        return ans;
    }

    private void dfs(int i) {
        if (i == digits.length) {
            ans.add(new String(path));
            return;
        }
        for (char c : MAPPING[digits[i] - '0'].toCharArray()) {
            path[i] = c; // 直接覆盖
            dfs(i + 1);
        }
    }
}

~~~



### 78.子集

![image-20240225101633728](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240225101633728.png)

<font color=#f00>**方法一：输入的角度，选或不选**</font>

~~~java
class Solution {
    private final List<List<Integer>> ans = new ArrayList<>();
    private final List<Integer> path = new ArrayList<>();
    private int[] nums;

    public List<List<Integer>> subsets(int[] nums) {
        this.nums = nums;
        dfs(0);
        return ans;
    }

    private void dfs(int i) {
        if (i == nums.length) {
            ans.add(new ArrayList<>(path)); // 固定答案
            return;
        }
        // 不选 nums[i]
        dfs(i + 1);
        // 选 nums[i]
        path.add(nums[i]);
        dfs(i + 1);
        path.remove(path.size() - 1); // 恢复现场
    }
}

~~~

<font color=#f00>**方法二：答案的角度（选哪个数）**</font>

![image-20240419102227459](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240419102227459.png)

~~~java
class Solution {
    private final List<List<Integer>> ans = new ArrayList<>();
    private final List<Integer> path = new ArrayList<>();
    private int[] nums;

    public List<List<Integer>> subsets(int[] nums) {
        this.nums = nums;
        dfs(0);
        return ans;
    }

    private void dfs(int i) {
        //这种写法每次递归都会得到一个res中的答案，所以每次开始递归的时候就要将path添加到答案中
        ans.add(new ArrayList<>(path)); // 固定答案
        if (i == nums.length) return;
        for (int j = i; j < nums.length; ++j) { // 枚举选择的数字
            path.add(nums[j]);
            dfs(j + 1);
            path.remove(path.size() - 1); // 恢复现场
        }
    }
}
~~~



### 131.分割回文串

![image-20240225101708031](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240225101708031.png)

~~~java
class Solution {
    private final List<List<String>> ans = new ArrayList<>();
    private final List<String> path = new ArrayList<>();
    private String s;

    public List<List<String>> partition(String s) {
        this.s = s;
        dfs(0, 0);
        return ans;
    }

    private boolean isPalindrome(int left, int right) {
        while (left < right)
            if (s.charAt(left++) != s.charAt(right--))
                return false;
        return true;
    }

    // start 表示当前这段回文子串的开始位置
    private void dfs(int i, int start) {
        if (i == s.length()) {
            ans.add(new ArrayList<>(path)); // 复制 path
            return;
        }

        // 不选 i 和 i+1 之间的逗号（i=n-1 时一定要选）
        if (i < s.length() - 1)
            dfs(i + 1, start);

        // 选 i 和 i+1 之间的逗号（把 s[i] 作为子串的最后一个字符）
        if (isPalindrome(start, i)) {
            path.add(s.substring(start, i + 1));
            dfs(i + 1, i + 1); // 下一个子串从 i+1 开始
            path.remove(path.size() - 1); // 恢复现场
        }
    }
}

~~~



~~~java
class Solution {
    private final List<List<String>> ans = new ArrayList<>();
    private final List<String> path = new ArrayList<>();
    private String s;

    public List<List<String>> partition(String s) {
        this.s = s;
        dfs(0);
        return ans;
    }

    private boolean isPalindrome(int left, int right) {
        while (left < right)
            if (s.charAt(left++) != s.charAt(right--))
                return false;
        return true;
    }
	//大于等于i的需要枚举
    private void dfs(int i) {
        if (i == s.length()) {
            ans.add(new ArrayList<>(path)); // 复制 path
            return;
        }
        for (int j = i; j < s.length(); ++j) { // 枚举子串的结束位置
            if (isPalindrome(i, j)) {
                path.add(s.substring(i, j + 1));
                dfs(j + 1);
                path.remove(path.size() - 1); // 恢复现场
            }
        }
    }
}

~~~





![image-20240225115922224](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240225115922224.png)

### 784.字母大小写全排列

![image-20240225121039399](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240225121039399.png)

<font color=#00f>**选或者不选**</font>

~~~java
class Solution {
    List<String> ans = new ArrayList<>();
    char[] path;

    public List<String> letterCasePermutation(String s) {
        path = s.toCharArray();
        dfs(0);
        return ans;
    }

    void dfs(int i) {
        if (i == path.length) {
            ans.add(new String(path));
            return;
        }
        //不选
        dfs(i + 1);
        //选(只有是字母才能选)
        if (path[i] >= 'a' && path[i] <= 'z') {
            path[i] = (char) (path[i] - 32);
            dfs(i + 1);
        } else if (path[i] >= 'A' && path[i] <= 'Z') {
            path[i] = (char) (path[i] + 32);
            dfs(i + 1);
        }
    }
}

~~~

<font color=#00f>**从答案的角度出发**</font>

~~~java
class Solution {
    List<String> ans = new ArrayList<>();
    char[] path;

    public List<String> letterCasePermutation(String s) {
        path = s.toCharArray();
        dfs(0);
        return ans;
    }

    void dfs(int i) {
        //固定答案
        ans.add(new String(path));
        //站在答案的角度，第i个变大的是谁
        // 比如第一次，i=0，那么就是判断哪一个字母第一次需要改变，其余的不变，所以这里需要恢复现场
        for (int j = i; j < path.length; j++) {
            //是数字，不能选
            if (path[j] >= '0' && path[j] <= '9') continue;
            //第i个改变的数字为j
            if (path[j] >= 'a' && path[j] <= 'z') {
                path[j] = (char) (path[j] - 32);
            } else if (path[j] >= 'A' && path[j] <= 'Z') {
                path[j] = (char) (path[j] + 32);
            }
            dfs(j + 1);
            //恢复现场
            if (path[j] >= 'a' && path[j] <= 'z') {
                path[j] = (char) (path[j] - 32);
            } else if (path[j] >= 'A' && path[j] <= 'Z') {
                path[j] = (char) (path[j] + 32);
            }
        }
    }
}

~~~



### ==1601.==<font color=#f00>**最多可达成的换楼请求数目（难度很高的题）**</font>

![image-20240226085517251](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240226085517251.png)

~~~java
class Solution {
    public int maximumRequests(int n, int[][] requests) {
        // 获取提出更换请求的员工个数
        int m = requests.length;
        // max 代表的是员工的状态 -----> 员工请求的请求可以被接受 / 不接受 每个员工有两种状态 那么就有 2^m 次种状态结果
        int maxStatus = 1 << m, res = 0;
        // 记录是否成功接受请求的数组
        int[] netChangeArr = new int[n];

        // 枚举所有状态，二进制中 1 的个数为当前状态的请求个数
        for (int i = 0; i < maxStatus; i++) {
            int[] tmp = new int[n];
            int state = i;
            // 用于记录当前的员工请求下标
            int idx = 0;
            // 用于记录当前状态下的接受的请求个数
            int cnt = 0;
            // 用于获取当前状态中二进制数字中1的个数
            while (state > 0) {
                // 判断最后一位是否为1
                int isAccept = state & 1;
                // 如果为 1 则说明了当前状态被接受了 需要记录住房情况
                if (isAccept == 1) {
                    // 获取需要从哪里 搬到 哪里的位置信息
                    int from = requests[idx][0];
                    int to = requests[idx][1];
                    // 记录住房净变化
                    tmp[from]--;
                    tmp[to]++;
                    cnt++;
                }
                // 获取前一个员工的状态请求情况
                state >>= 1;
                idx++;
            }
            // 根据我们上面完成得到记录住房变化 需要满足每栋楼员工净变化为 0
            // 滚动获取最大值
            if (Arrays.equals(tmp, netChangeArr)) {
                res = Math.max(res,cnt);
            }
        }
        return res;
    }
}
~~~



### ==2397.==被列覆盖的最多行数

![image-20240226090451650](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240226090451650.png)

![image-20240226090509349](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240226090509349.png)

~~~java
class Solution {
    public int maximumRows(int[][] mat, int numSelect) {
        int m = mat.length, n = mat[0].length;
        int[] mask = new int[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                //处理原矩阵
                //将对应位置的数左移相应的数量之后相加
                mask[i] |= mat[i][j] << j;
            }
        }

        int ans = 0;
        for (int subset = 0; subset < (1 << n); subset++) {
            //位运算中的1的个数代表集合中的元素个数，如果1的个数等于能选择的数字长度就记录
            if (Integer.bitCount(subset) == numSelect) {
                int coveredRows = 0;
                for (int row : mask) {
                    //位运算&之后如果等于自身的话，表示当前选择能进行覆盖当前行
                    if ((row & subset) == row) {
                        coveredRows++;
                    }
                }
                ans = Math.max(ans, coveredRows);
            }
        }
        return ans;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/maximum-rows-covered-by-columns/solutions/1798794/by-endlesscheng-dvxe/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

[集合论到位运算， 常见位运算技巧分类总结](https://leetcode.cn/circle/discuss/CaOJ45/)

### ==306.==累加数==（高精度算法）==

![image-20240226100536044](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240226100536044.png)

![image-20240419105341580](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240419105341580.png)

~~~java
class Solution {
    String num;
    int n;
    List<List<Integer>> list = new ArrayList<>();
    public boolean isAdditiveNumber(String _num) {
        num = _num;
        n = num.length();
        return dfs(0);
    }
    boolean dfs(int u) {
        int m = list.size();
        if (u == n) return m >= 3;//题目描述至少包含三个数，看是否满足要求
        int max = num.charAt(u) == '0' ? u + 1 : n;//处理前导0
        List<Integer> cur = new ArrayList<>();
        //枚举[u,n-1]
        for (int i = u; i < max; i++) {
            cur.add(0, num.charAt(i) - '0');
            if (m < 2 || check(list.get(m - 2), list.get(m - 1), cur)) {
                list.add(cur);
                if (dfs(i + 1)) return true;
                list.remove(list.size() - 1);
            }
        }
        return false;
    }
    //高精度加法，逆序存放数值，检查a+b是否等于c
    boolean check(List<Integer> a, List<Integer> b, List<Integer> c) {
        //高精度算法部分
        List<Integer> ans = new ArrayList<>();
        int t = 0;
        for (int i = 0; i < a.size() || i < b.size(); i++) {
            if (i < a.size()) t += a.get(i);
            if (i < b.size()) t += b.get(i);
            ans.add(t % 10);
            t /= 10;
        }
        if (t > 0) ans.add(t);
        //检查是否相等部分
        boolean ok = c.size() == ans.size();
        for (int i = 0; i < c.size() && ok; i++) {
            if (c.get(i) != ans.get(i)) ok = false;
        }
        return ok;
    }
}

~~~



### 2698.求一个整数的惩罚数



## 十五.组合型回溯|剪枝

<font color=#00f>**相较于排列型回溯，组合型回溯有更多的剪枝方法**</font>

### ==77.==组合

![image-20240227091923948](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227091923948.png)

~~~java
class Solution {
    private int k;
    private final List<Integer> path = new ArrayList<>();
    private final List<List<Integer>> ans = new ArrayList<>();

    public List<List<Integer>> combine(int n, int k) {
        this.k = k;
        dfs(n);
        return ans;
    }

    //倒序枚举是为了让不等式更加方便
    private void dfs(int i) {
        int d = k - path.size(); // 还要选 d 个数
        if (d == 0) {
            ans.add(new ArrayList<>(path));
            return;
        }
        for (int j = i; j >= d; --j) {
            path.add(j);
            dfs(j - 1);
            path.remove(path.size() - 1);
        }
    }
}

~~~



~~~java
class Solution {
    private int k;
    private final List<Integer> path = new ArrayList<>();
    private final List<List<Integer>> ans = new ArrayList<>();

    public List<List<Integer>> combine(int n, int k) {
        this.k = k;
        dfs(n);
        return ans;
    }

    private void dfs(int i) {
        int d = k - path.size(); // 还要选 d 个数
        if (d == 0) {
            ans.add(new ArrayList<>(path));
            return;
        }
        // 不选 i
        if (i > d) dfs(i - 1);
        // 选 i
        path.add(i);
        dfs(i - 1);
        path.remove(path.size() - 1);
    }
}

~~~



### ==216.==组合总数III

![image-20240227093221607](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227093221607.png)

~~~java
class Solution {
    private int k;
    private final List<Integer> path = new ArrayList<>();
    private final List<List<Integer>> ans = new ArrayList<>();

    public List<List<Integer>> combinationSum3(int k, int n) {
        this.k = k;
        dfs(9, n);
        return ans;
    }

    private void dfs(int i, int t) {
        int d = k - path.size(); // 还要选 d 个数
        if (t < 0 || t > (i * 2 - d + 1) * d / 2) // 剪枝
            return;
        if (d == 0) {
            ans.add(new ArrayList<>(path));
            return;
        }
        for (int j = i; j >= d; --j) {
            path.add(j);
            dfs(j - 1, t - j);
            path.remove(path.size() - 1);
        }
    }
}

~~~

~~~java
class Solution {
    private int k;
    private final List<Integer> path = new ArrayList<>();
    private final List<List<Integer>> ans = new ArrayList<>();

    public List<List<Integer>> combinationSum3(int k, int n) {
        this.k = k;
        dfs(9, n);
        return ans;
    }

    private void dfs(int i, int t) {
        int d = k - path.size(); // 还要选 d 个数
        if (t < 0 || t > (i * 2 - d + 1) * d / 2) // 剪枝
            return;
        if (d == 0) {
            ans.add(new ArrayList<>(path));
            return;
        }
        // 不选 i
        if (i > d) dfs(i - 1, t);
        // 选 i
        path.add(i);
        dfs(i - 1, t - i);
        path.remove(path.size() - 1);
    }
}

~~~



### ==22.==括号生成：

![image-20240227100311607](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227100311607.png)

<font color=#00f>**运用选或不选的思路，一共是2n个位置，左括号必须先放，**</font>

~~~java
class Solution {
    private int n;
    private char[] path;
    private final List<String> ans = new ArrayList<>();

    public List<String> generateParenthesis(int n) {
        this.n = n;
        path = new char[n * 2];
        dfs(0, 0);
        return ans;
    }

    private void dfs(int i, int open) {
        if (i == n * 2) {
            ans.add(new String(path));
            return;
        }
        if (open < n) { // 可以填左括号，左括号的个数没达到n个，达到n个之后只能放右括号
            path[i] = '(';
            dfs(i + 1, open + 1);
        }
        if (i - open < open) { // 可以填右括号，右括号的数目小于左括号的数目
            path[i] = ')';
            dfs(i + 1, open);
        }
    }
}

~~~



~~~java
class Solution {
    private int n;
    private final List<Integer> path = new ArrayList<>();
    private final List<String> ans = new ArrayList<>();

    public List<String> generateParenthesis(int n) {
        this.n = n;
        dfs(0, 0);
        return ans;
    }

    // balance = 左括号个数 - 右括号个数
    private void dfs(int i, int balance) {
        if (path.size() == n) {
            char[] s = new char[n * 2];
            Arrays.fill(s, ')');
            for (int j : path) s[j] = '(';
            ans.add(new String(s));
            return;
        }
        // 可以填 0 到 balance 个右括号
        for (int close = 0; close <= balance; ++close) { // 填 close 个右括号
            path.add(i + close); // 填 1 个左括号
            dfs(i + close + 1, balance - close + 1);
            path.remove(path.size() - 1);
        }
    }
}

~~~



### ==301.==删除无效的括号

![image-20240227101517376](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227101517376.png)

~~~java
class Solution {
    Set<String> set = new HashSet<>();
    // n：字符串长度；max：最大括号数（单括号）；maxPathLen：记录「爆搜」过程中的最大路径子串的长度
    int n, max, maxPathLen;
    String s;

    public List<String> removeInvalidParentheses(String _s) {
        s = _s;
        n = s.length();
        int left = 0, right = 0;

        // 统计多余的括号数量
        for (char c : s.toCharArray()) {
            if (c == '(') left++;
            else if (c == ')') {
                if (left != 0) left--;
                else right++;
            }
        }
        maxPathLen = n - left - right;      // 提前更新 maxPathLen

        // 统计左右括号数量
        int left2 = 0, right2 = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') left2++;
            else if (c == ')') right2++;
        }

        max = Math.min(left2, right2);
        dfs(0, "", left, right, 0);
        return new ArrayList<>(set);    // 将Set集合转为List返回
    }

    /**
     * 遍历 _s 字符串，记录有效路径
     * @param curCharIndex 当前遍历的字符下标
     * @param path 遍历时的路径（括号组合字符串）
     * @param left 多余的左括号数量
     * @param right 多余的右括号数量
     * @param score 分数，用于标记左右括号的得分
     */
    private void dfs(int curCharIndex, String path, int left, int right, int score) {
        // 剪枝：合法路径的得分范围为 0 <= score <= max；多余的括号数量为负数，说明删多了，不符合
        if (left < 0 || right < 0 || score < 0 || score > max) return;

        if (left == 0 && right == 0) {
            // 多余的括号为0，且当前路径长度等于最大路径子串的长度，则符合
            if (path.length() == maxPathLen) {
                set.add(path);
            }
        }

        if (curCharIndex == n) return;      // 搜索完毕，退出（放在此处是为了记录完最后一个字符）

        char c = s.charAt(curCharIndex);     // 获取当前字符

        // 每一种选择都对应 添加/不添加
        if (c == '(') {         // 添加左括号，score + 1；不添加score不变，多余的左括号数量-1
            dfs(curCharIndex + 1, path + c, left, right, score+ 1);
            dfs(curCharIndex + 1, path, left - 1, right, score);
        } else if (c == ')') {      // 添加右括号，score - 1；不添加score不变，多余的右括号数量-1
            dfs(curCharIndex + 1, path + c, left, right, score - 1);
            dfs(curCharIndex + 1, path, left, right - 1, score);
        } else {        // 普通字符，score不变
            dfs(curCharIndex + 1, path + c, left, right, score);
        }
    }
}
~~~

<font color=#00f>**自己写的逻辑，卡在不知道怎么将List<Character>转化为字符串**</font>

~~~java
class Solution {
    private int minNum = Integer.MAX_VALUE;// 记录最少的删除次数
    private int num = 0;
    private int n;
    List<Character> path = new ArrayList<>();
    List<String> ans = new ArrayList<>();
    char[] sChar;

    public List<String> removeInvalidParentheses(String s) {
        n = s.length();
        sChar = s.toCharArray();
        dfs(0, 0);
    }

    // i代表第几个字符需要进行遍历
    private void dfs(int balance, int i) {
        if (i == n) {
            if (num == minNum) {
                ans.add(path.stream().map(Object::toString).collect(Collections.joining()));
                num = 0;
            }
            if (num < minNum) {
                minNum = num;
                ans.clear();
                ans.add(path.stream().map(Object::toString).collect(Collections.joining()));
                num = 0;
            }
            return;
        }
        if (s[i] != '(' || s[i] != ')') {
            dfs(balance, i + 1);
        }
        if (s[i] == '(') {
            path.add('(');
            dfs(balance + 1, i + 1);
            path.remove(path.size() - 1);
        }
        if (s[i] == ')' && balance > 0) {
            path.add(')');
            dfs(balance - 1, i + 1);
            path.remove(path.size() - 1);
        } else if (s[i] == ')' && balance == 0) {
            dfs(balance, i + 1);
            num += 1;
        }
    }
}
~~~



## 十六.排列型回溯

### 46.全排列（==创建指定长度的集合==）

![image-20240227110237255](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227110237255.png)

~~~java
class Solution {
    private int[] nums;
    private List<Integer> path;
    private boolean[] onPath;
    private final List<List<Integer>> ans = new ArrayList<>();

    public List<List<Integer>> permute(int[] nums) {
        this.nums = nums;
        path = Arrays.asList(new Integer[nums.length]);
        onPath = new boolean[nums.length];
        dfs(0);
        return ans;
    }

    private void dfs(int i) {
        if (i == nums.length) {
            ans.add(new ArrayList<>(path));
            return;
        }
        for (int j = 0; j < nums.length; ++j) {
            if (!onPath[j]) {
                path.set(i, nums[j]);
                onPath[j] = true;
                dfs(i + 1);
                onPath[j] = false; // 恢复现场
            }
        }
    }
}

~~~



### ==51.==N皇后

![image-20240227145442811](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227145442811.png)

<font color=#00f>**放皇后从不能同行不能同列的要求，可以推出每一行每一列只有一个皇后，所以创建一个col数组记录每行的皇后放在第几列，就是一个n-1的全排列，但是，要求二是不能在同一个斜线，所以还需要对这些全排列进行剪枝，得到剪枝后的才是真正的可以存在的解法**</font>

<font color=#00f>**不能在同一斜线的剪枝操作：由于是从上到下进行枚举，所以只需要判断当前位置的右上和左上是否有皇后，如果有当前位置不满足要求进行剪枝；如何判断是否在右上方：右上方的荒兽行号加列号是一样的，左上方的行号减列号是一样的**</font>

![image-20240227112939989](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227112939989.png)

~~~java
class Solution {
    private int n;
    private int[] col;
    private boolean[] onPath, diag1, diag2;
    private final List<List<String>> ans = new ArrayList<>();

    public List<List<String>> solveNQueens(int n) {
        this.n = n;
        col = new int[n];
        onPath = new boolean[n];
        diag1 = new boolean[n * 2 - 1];
        diag2 = new boolean[n * 2 - 1];
        dfs(0);
        return ans;
    }
	
    private void dfs(int r) {
        if (r == n) {
            //遍历到第n行时将当前情况放入到结果中
            List<String> board = new ArrayList<>(n);
            for (int c : col) {
                char[] row = new char[n];
                Arrays.fill(row, '.');
                row[c] = 'Q';
                board.add(new String(row));
            }
            ans.add(board);
            return;
        }
        for (int c = 0; c < n; ++c) {
            int rc = r - c + n - 1;
            if (!onPath[c] && !diag1[r + c] && !diag2[rc]) {
                col[r] = c;
                onPath[c] = diag1[r + c] = diag2[rc] = true;
                dfs(r + 1);
                onPath[c] = diag1[r + c] = diag2[rc] = false; // 恢复现场
            }
        }
    }
}

~~~



### 52.N皇后II

![image-20240227145420301](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227145420301.png)

~~~java
class Solution {
    private int n, ans;
    private boolean[] onPath, diag1, diag2;

    public int totalNQueens(int n) {
        this.n = n;
        onPath = new boolean[n];
        diag1 = new boolean[n * 2 - 1];
        diag2 = new boolean[n * 2 - 1];
        dfs(0);
        return ans;
    }

    private void dfs(int r) {
        if (r == n) {
            ans++;
            return;
        }
        for (int c = 0; c < n; ++c) {
            int rc = r - c + n - 1;
            if (!onPath[c] && !diag1[r + c] && !diag2[rc]) {
                onPath[c] = diag1[r + c] = diag2[rc] = true;
                dfs(r + 1);
                onPath[c] = diag1[r + c] = diag2[rc] = false; // 恢复现场
            }
        }
    }
}
~~~

# <font color=#f00 size= 7>动态规划：</font>

动态规划常见题型：

- 动态规划基础
- 背包问题
- 打家劫舍
- 股票问题
- 子序列问题
- <span style="background-color:#eeeeee;">区间dp，和概率dp都是竞赛中的难题，大厂面试题一般不会上升到这个难度</span>

<font color=#f00>**解决动态规划必须要思考的点（动归五部曲）**</font>

- 明白dp数组的定义以及下标的含义
- 递推公式
- dp数组如何初始化
- 遍历顺序（先遍历背包还是先遍历物品）
- 打印dp数组

### 509.<u>斐波那契数</u>(ak，不用复习)

![image-20240229134534791](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240229134534791.png)

- dp[i]:第i个斐波那契数的值为dp[i];
- 递推公式：dp[i] = dp[i-1] + dp[i-2];
- dp数组如何初识化：dp[0] = 1,dp[1] =1;
- 遍历顺序：因为后面的数都是从前面的数获得的，所以只能是从前往后遍历

~~~java
class Solution {
    public int fib(int n) {
        if(n==0||n==1) return n;
        
        int[] dp = new int[n+1];
        dp[0]=0;
        dp[1]=1;
        for(int i = 2; i <= n ;i++){
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
}
~~~

### 62.<u>不同路径</u>（ak，不用复习）

![image-20240229140427347](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240229140427347.png)

- dp[i] [j]:从（0，0）到（i，j）有多少种不同的路径
- 递推公式：dp[i] [j] = dp[i-1] [j] + dp[i] [j-1];
- 初始化：第一列对应的位置为i，第一行对应的位置为j;

~~~java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i <= m - 1; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i <= n - 1; i++) {
            dp[0][i] = 1;
        }

        for (int i = 1; i <= m - 1; i++) {
            for (int j = 1; j <= n - 1; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }
}
~~~

### 63.不同路径II

![image-20240229142334541](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240229142334541.png)

- dp[i] [j]到达(i,j)对应位置路径的数量为dp[i] [j]
- 递推公式： 先要进行判断，想要求的位置是是障碍，如果是障碍将当前位置的值设置为1
- 初始化：还是第一行和第一列初始化为1，但是要注意，如果第一行和第一列有障碍需要将障碍后面的位置都设置为0.

~~~java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;

        int[][] dp = new int[m][n];
        for (int i = 0; i <= m - 1; i++) {
            if(obstacleGrid[i][0] == 1){break;}
            dp[i][0] = 1;
        }
        for (int i = 0; i <= n - 1; i++) {
            if(obstacleGrid[0][i] == 1){break;}
            dp[0][i] = 1;
        }

        for (int i = 1; i <= m - 1; i++) {
            for (int j = 1; j <= n - 1; j++) {
                if(obstacleGrid[i][j] ==1){
                    dp[i][j] = 0;
                }else{
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
                
            }
        }
        return dp[m - 1][n - 1];
    }
}
~~~

### ==343.==整数拆分

![image-20240229143644377](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240229143644377.png)

- dp[i]对i进行拆分，得到的最大乘积为dp[i]
- 递推公式：dp[i] = max(dp[i-j] * j)<font color=#f00>**这道题递推公式比较难想**</font>
- 初始化：dp[0]=0, dp[1]=0, dp[2]=1;

~~~java
class Solution {
    public int integerBreak(int n) {
        //dp[i] 为正整数 i 拆分后的结果的最大乘积
        int[] dp = new int[n+1];
        dp[2] = 1;
        for(int i = 3; i <= n; i++) {
            for(int j = 1; j <= i-j; j++) {
                // 这里的 j 其实最大值为 i-j,再大只不过是重复而已，
                //并且，在本题中，我们分析 dp[0], dp[1]都是无意义的，
                //j 最大到 i-j,就不会用到 dp[0]与dp[1]
                dp[i] = Math.max(dp[i], Math.max(j*(i-j), j*dp[i-j]));
                // j * (i - j) 是单纯的把整数 i 拆分为两个数 也就是 i,i-j ，再相乘
                //而j * dp[i - j]是将 i 拆分成两个以及两个以上的个数,再相乘。
            }
        }
        return dp[n];
    }
}

~~~

### 96.不同的二叉搜索树

![image-20240229145127875](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240229145127875.png)

<font color=#00f>**这道题主要就是找规律,自己把规律找明白之后就能写出来了**</font>

- dp[i]:整数为i的二叉搜素树的个数为dp[i]个
- dp[i] += dp[j] * dp[i-1-j] for i in range(n);

~~~java
class Solution {
    public int numTrees(int n) {
        int[] dp = new int[n+1];
        dp[0]=1;
        dp[1]=1;
        for(int i = 2; i <=n ; i++){
            for(int j = 0; j<= i-1;j++){
                dp[i] += (dp[j] * dp[i-1-j]);
            }
        }
        return dp[n];
    }
}
~~~

### 120.三角形的最小路径和(ak，不用复习)

![image-20240411205356121](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240411205356121.png)

~~~java
class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        // dp数组的含义以i行j个处最小的路径和
        int[][] dp = new int[n + 1][n + 1];
        // 初始化，因为是求最小路径和，所以将不合法的地方初始化为正无穷
        for (int[] d : dp) {
            Arrays.fill(d, Integer.MAX_VALUE);
        }
        dp[0][0] = triangle.get(0).get(0);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (j == 0) {
                    dp[i][j] = dp[i - 1][j] + triangle.get(i).get(j);
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle.get(i).get(j);
                }
            }
        }
        int res = Integer.MAX_VALUE;
        for(int num:dp[n-1]){
            if(num < res){
                res = num;
            }
        }
        return res;
    }
}
~~~

### 64.最小路径和（ak，不用复习）

![image-20240411210526302](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240411210526302.png)

~~~java
class Solution {
    public int minPathSum(int[][] grid) {
        // 获取m行n列
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int[] d : dp) {
            Arrays.fill(d, Integer.MAX_VALUE);
        }
        // 初始化第一行和第一列的元素
        dp[0][0] = grid[0][0];
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }
}
~~~



## 十七.动态规划|记忆化搜素到递推

### 198.打家劫舍（ak）

<font color=#00f>**使用递归的方法写：由于使用搜素树是指数级别的所以会超时**</font>

~~~java
class Solution {
    int[] nums;

    public int rob(int[] nums) {
        this.nums = nums;
        return dfs(nums.length - 1);

    }

    private int dfs(int i) {
        if (i < 0) {
            return 0;
        }
        return Math.max(dfs(i - 1), dfs(i - 2) + nums[i]);
    }
}
~~~

<font color=#00f>**记忆化搜素：时间复杂度优化到了O(n)**</font>

~~~java
class Solution {
    int[] nums;
    int[] cache;

    public int rob(int[] nums) {
        this.nums = nums;
        int n = nums.length;
        cache = new int[n];
        Arrays.fill(cache, -1);
        return dfs(nums.length - 1);
    }

    private int dfs(int i) {
        if (i < 0) {
            return 0;
        }
        if (cache[i] != -1) {
            return cache[i];
        }
        int res = Math.max(dfs(i - 1), dfs(i - 2) + nums[i]);
        cache[i] = res;
        return res;
    }
}
~~~

<font color=#00f>**递推方法**</font>

~~~java
class Solution {
    int[] f;

    public int rob(int[] nums) {
        int n = nums.length;
        f = new int[n + 2];
        Arrays.fill(f, 0);
        for (int i = 0; i < n; i++) {
            f[i + 2] = Math.max(f[i + 1], f[i] + nums[i]);
        }
        return f[n + 1];
    }
}
~~~



### 70.爬楼梯

![image-20240227200825609](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227200825609.png)

- dp[i]:达到第i阶有dp[i]种方法
- 递推公式：dp[i] = dp[i-2] + dp[i-1];
- 初始化：dp[0],因为第0阶台阶没有任何意义，所以可以从dp[1] = 1、dp[2]=2 进行初始化
- 遍历顺序：从前向后

<font color=#00f>**记忆化搜素**</font>

~~~java
class Solution {
    int[] cache;

    public int climbStairs(int n) {
        cache = new int[n + 1];
        return dfs(n);
    }
	//达到第i阶有几种方法
    private int dfs(int i) {
        if (i <= 1) {
            return 1;
        }
        if (cache[i] != 0) {
            return cache[i];
        }
        int res = dfs(i - 1) + dfs(i - 2);
        cache[i] = res;
        return res;
    }
}
~~~

<font color=#00f>**递推**</font>

~~~java
class Solution {
    public int climbStairs(int n) {
        int[] f = new int[n + 2];
        f[0] = 1;
        f[1] = 1;
        for (int i = 2; i <= n; i++) {
            f[i] = f[i - 1] + f[i - 2];
        }
        return f[n];
    }
}
~~~



### 746.使用最小花费爬楼梯

![image-20240229135907566](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240229135907566.png)

- dp[i]:达到第i阶需要的最小花费
- dp[i]=min(dp[i-1] + cost[i-1],dp[i-2]+cost[i-2])
- 初识化：dp[0]=0 ;dp[1] = 0;
- 顺序：从前向后

~~~java
public class Solution {
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] f = new int[n + 1];
        for (int i = 2; i <= n; i++) {
            f[i] = Math.min(f[i - 1] + cost[i - 1], f[i - 2] + cost[i - 2]);
        }
        return f[n];
    }
}
~~~

### ==2466.==统计构造好字符串的方案数

![image-20240228085521017](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240228085521017.png)

~~~java
class Solution {
    public int countGoodStrings(int low, int high, int zero, int one) {
        //跟爬楼梯的做法一模一样，不过就想当与是多次爬楼梯的结果的集合，落到low和high区间都是解
        //定义f[i]表示长为i的字符串个数
        //f[i] = f[i-zero] + f[i-one]
        //f[0] = 1 因为当字符串为空的时候有一个解
        final int MOD = (int) 1e9 + 7;
        int ans = 0;
        var f = new int[high + 1]; // f[i] 表示构造长为 i 的字符串的方案数
        f[0] = 1; // 构造空串的方案数为 1
        for (int i = 1; i <= high; i++) {
            if (i >= one) f[i] = (f[i] + f[i - one]) % MOD;
            if (i >= zero) f[i] = (f[i] + f[i - zero]) % MOD;
            if (i >= low) ans = (ans + f[i]) % MOD;
        }
        return ans;
    }
}
~~~



### 213.打家劫舍II

![image-20240228094211401](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240228094211401.png)

~~~java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        return Math.max(nums[0] + rob1(nums, 2, n - 1), rob1(nums, 1, n));
    }
    
    // 198. 打家劫舍
    private int rob1(int[] nums, int start, int end) { // [start,end) 左闭右开
        int f0 = 0, f1 = 0;
        for (int i = start; i < end; ++i) {
            int newF = Math.max(f1, f0 + nums[i]);
            f0 = f1;
            f1 = newF;
        }
        return f1;
    }
}

~~~

<font color=#f00 size= 6>**买卖股票经典例题**</font>

### 121.买卖股票的最佳时机

![image-20240303161957317](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240303161957317.png)

- dp数组的含义：dp[i] [0]第i天不持有股票，我们所能得到的最大金额；dp[i] [1]第i天不持有股票，我们所能得到的最大金额；

~~~java
//>贪心做法：
class Solution {
    public int maxProfit(int[] prices) {
        // 找到一个最小的购入点
        int low = Integer.MAX_VALUE;
        // res不断更新，直到数组循环完毕
        int res = 0;
        for(int i = 0; i < prices.length; i++){
            low = Math.min(prices[i], low);
            res = Math.max(prices[i] - low, res);
        }
        return res;
    }
}

// > 动态规划：版本一
// 解法1
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) return 0;
        int length = prices.length;
        // dp[i][0]代表第i天持有股票的最大收益
        // dp[i][1]代表第i天不持有股票的最大收益
        int[][] dp = new int[length][2];
        int result = 0;
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for (int i = 1; i < length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], -prices[i]);
            dp[i][1] = Math.max(dp[i - 1][0] + prices[i], dp[i - 1][1]);
        }
        return dp[length - 1][1];
    }
}

// > 动态规划：版本二
class Solution {
  public int maxProfit(int[] prices) {
    int[] dp = new int[2];
    // 记录一次交易，一次交易有买入卖出两种状态
    // 0代表持有，1代表卖出
    dp[0] = -prices[0];
    dp[1] = 0;
    // 可以参考斐波那契问题的优化方式
    // 我们从 i=1 开始遍历数组，一共有 prices.length 天，
    // 所以是 i<=prices.length
    for (int i = 1; i <= prices.length; i++) {
      // 前一天持有；或当天买入
      dp[0] = Math.max(dp[0], -prices[i - 1]);
      // 如果 dp[0] 被更新，那么 dp[1] 肯定会被更新为正数的 dp[1]
      // 而不是 dp[0]+prices[i-1]==0 的0，
      // 所以这里使用会改变的dp[0]也是可以的
      // 当然 dp[1] 初始值为 0 ，被更新成 0 也没影响
      // 前一天卖出；或当天卖出, 当天要卖出，得前一天持有才行
      dp[1] = Math.max(dp[1], dp[0] + prices[i - 1]);
    }
    return dp[1];
  }
}

~~~

### 122.买卖股票的最佳时机II

![image-20240303170100687](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240303170100687.png)

- dp数组的含义：dp[i] [0]第i天持有股票，我们所能得到的最大金额；dp[i] [1]第i天不持有股票，我们所能得到的最大金额；

<font color=#f00>**两道题唯一的区别就是，多次买卖是从持有状态变为卖出状态，单次买卖肯定是从0开始买入股票**</font>

~~~java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0)
            return 0;
        int length = prices.length;
        // dp[i][0]代表第i天持有股票的最大收益
        // dp[i][1]代表第i天不持有股票的最大收益
        int[][] dp = new int[length][2];
        int result = 0;
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for (int i = 1; i < length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
            dp[i][1] = Math.max(dp[i - 1][0] + prices[i], dp[i - 1][1]);
        }
        return dp[length - 1][1];
    }
}
~~~

### 123.买卖股票的最佳时机III

![image-20240303170611407](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240303170611407.png)

~~~java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0)
            return 0;
        int length = prices.length;
        // dp[i][0]代表第i天不操作
        // dp[i][1]代表第i天第一次持有
        // dp[i][2]代表第i天第一次不持有
        // dp[i][3]代表第i天第二次持有
        // dp[i][4]代表第i天第二次不持有
        int[][] dp = new int[length][5];
        int result = 0;
        dp[0][1] = -prices[0];
        // 初始化第二次买入的状态是确保 最后结果是最多两次买卖的最大利润
        dp[0][3] = -prices[0];//可以理解成当天买了又卖又买
        for (int i = 1; i < length; i++) {
            dp[i][0] = dp[i - 1][0];
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
            dp[i][4] = Math.max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
            
        }
        return dp[length -1][4];
    }
}
~~~



~~~java
// 版本一
class Solution {
    public int maxProfit(int[] prices) {
        int len = prices.length;
        // 边界判断, 题目中 length >= 1, 所以可省去
        if (prices.length == 0) return 0;

        /*
         * 定义 5 种状态:
         * 0: 没有操作, 1: 第一次买入, 2: 第一次卖出, 3: 第二次买入, 4: 第二次卖出
         */
        int[][] dp = new int[len][5];
        dp[0][1] = -prices[0];
        // 初始化第二次买入的状态是确保 最后结果是最多两次买卖的最大利润
        dp[0][3] = -prices[0];

        for (int i = 1; i < len; i++) {
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i][1] + prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i][2] - prices[i]);
            dp[i][4] = Math.max(dp[i - 1][4], dp[i][3] + prices[i]);
        }

        return dp[len - 1][4];
    }
}

// 版本二: 空间优化
class Solution {
    public int maxProfit(int[] prices) {
        int[] dp = new int[4]; 
        // 存储两次交易的状态就行了
        // dp[0]代表第一次交易的买入
        dp[0] = -prices[0];
        // dp[1]代表第一次交易的卖出
        dp[1] = 0;
        // dp[2]代表第二次交易的买入
        dp[2] = -prices[0];
        // dp[3]代表第二次交易的卖出
        dp[3] = 0;
        for(int i = 1; i <= prices.length; i++){
            // 要么保持不变，要么没有就买，有了就卖
            dp[0] = Math.max(dp[0], -prices[i-1]);
            dp[1] = Math.max(dp[1], dp[0]+prices[i-1]);
            // 这已经是第二次交易了，所以得加上前一次交易卖出去的收获
            dp[2] = Math.max(dp[2], dp[1]-prices[i-1]);
            dp[3] = Math.max(dp[3], dp[2]+ prices[i-1]);
        }
        return dp[3];
    }
}

作者：代码随想录
链接：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/solutions/552849/123-mai-mai-gu-piao-de-zui-jia-shi-ji-ii-zfh9/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

### 188.买卖股票的最佳时机IV

![image-20240303173708255](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240303173708255.png)

~~~java
class Solution {
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        int m = 2 * k + 1;
        int[][] dp = new int[n][m];
        // 初始化第一列
        for (int j = 1; j < m; j += 2) {
            dp[0][j] = -prices[0];
        }

        // 0不操作，奇数天买入，偶数天卖出
        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= m - 2; j += 2) {
                dp[i][j + 1] = Math.max(dp[i - 1][j + 1], dp[i - 1][j] - prices[i]);
                dp[i][j + 2] = Math.max(dp[i - 1][j + 2], dp[i - 1][j + 1] + prices[i]);
            }
        }
        return dp[n - 1][m - 1];
    }
}
~~~



### 714.买卖股票的最佳时机含手续费

![image-20240303173951059](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240303173951059.png)

~~~java
class Solution {
    public int maxProfit(int[] prices, int fee) {
        if (prices == null || prices.length == 0)
            return 0;
        int length = prices.length;
        // dp[i][0]代表第i天持有股票的最大收益
        // dp[i][1]代表第i天不持有股票的最大收益
        int[][] dp = new int[length][2];
        int result = 0;
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for (int i = 1; i < length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
            dp[i][1] = Math.max(dp[i - 1][0] + prices[i] - fee, dp[i - 1][1]);
        }
        return dp[length - 1][1];
    }
}
~~~



### 309.买卖股票的最佳时机含冷冻期

![image-20240303174006370](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240303174006370.png)

~~~java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] f = new int[n + 2][2];
        f[1][1] = Integer.MIN_VALUE;
        for (int i = 0; i < n; ++i) {
            f[i + 2][0] = Math.max(f[i + 1][0], f[i + 1][1] + prices[i]);
            f[i + 2][1] = Math.max(f[i + 1][1], f[i][0] - prices[i]);
        }
        return f[n + 1][0];
    }
}

~~~

~~~java
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (n == 0) return 0;
        vector<vector<int>> dp(n, vector<int>(4, 0));
        dp[0][0] -= prices[0]; // 持股票
        for (int i = 1; i < n; i++) {
            dp[i][0] = max(dp[i - 1][0], max(dp[i - 1][3] - prices[i], dp[i - 1][1] - prices[i]));
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][3]);
            dp[i][2] = dp[i - 1][0] + prices[i];
            dp[i][3] = dp[i - 1][2];
        }
        return max(dp[n - 1][3], max(dp[n - 1][1], dp[n - 1][2]));
    }
};

~~~

<font color=#f00 size=6>**子序列系列动态规划例题**</font>

### ==300.==最长递增子序列（ak）

![image-20240303190926707](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240303190926707.png)

<font color=#00f>**感觉有点像双指针类型的例题**</font>

- dp数组的含义：以nums[i]为结尾的最长递增子序列的长度
- dp[i] = max(dp[j] + 1, dp[i])
- dp[1] = 1;
- i从小到大进行遍历，j可以从小到大或者从大到小都可以
- 最后输出，是以每个元素为结尾的最大值进行遍历，返回最大的值

~~~java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        //初始化
        Arrays.fill(dp,1);
        
        //遍历递推
        for(int i = 1; i <nums.length; i++){
            for(int j = 0; j<i; j++){
                if(nums[j] < nums[i]){
                    //解释为什么这个地方需要说明
                    dp[i] = Math.max(dp[j] + 1, dp[i]);
                }
            }
        }
        int res = 0;
        //遍历搜素结果
        for(int i = 0; i < nums.length; i++){
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}
~~~

### 674.最长连续递增序列（ak，不用复习）

![image-20240303192443731](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240303192443731.png)

- dp数组含义：以nums[i]为结尾的连续递增子序列的元素个数为dp[i]
- dp[i] = max(dp[j] + 1, dp[i])

~~~java
class Solution {
    public int findLengthOfLCIS(int[] nums) {
        int[] dp = new int[nums.length];
        // 初始化
        Arrays.fill(dp, 1);

        // 遍历递推
        for (int i = 1; i < nums.length; i++) {
            if (nums[i-1] < nums[i]) {
                // 解释为什么这个地方,因为要比较的只有前面一个元素，所以就比较i和i-1
                dp[i] = dp[i-1] + 1;
            }
        }
        int res = 0;
        // 遍历搜素结果
        for (int i = 0; i < nums.length; i++) {
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}
~~~

### 718.最长重复子数组

![image-20240303194107823](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240303194107823.png)

<font color=#00f>**本题难点：要求所求的子数组是连续的，如何通过dp数组将本题的状态进行保存**</font>

- dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的最长重复子数组的长度

- ~~~java
  if(nums[i-1] == nums[j-1]){
      dp[i][j] = dp[i-1][j-1] + 1;
  }
  ~~~

- 初始化：第一列和第一行都是没有意义的状态，所以将其初始化为0；其他的位置初始化什么都可以

- 遍历顺序：先遍历nums1和nums2都可以，遍历顺序都是从小到大

~~~java
class Solution {
    public int findLength(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        // dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的最长重复子数组的长度
        int[][] dp = new int[n1 + 1][n2 + 1];
        // 初始化：由于第一行第一列都没有意义，所以全都初始为0
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
            }
        }
        int ans = 0;
        for(int[] arr : dp){
            for(int ar : arr){
                ans = Math.max(ar, ans);
            }
        }
        return ans;
    }
}
~~~

### ==1143.==最长公共子序列

![image-20240304085405451](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304085405451.png)

<font color=#00f>**这道题相较于上一题不要求连续**</font>

<font color=#f00 size=4>**想不明白这个递推公式，现在写应该能ak**</font>

- ~~~java
  //dp[i][j]数组的含义：以nums[i-1]结尾和以nums[j-1]结尾的最长公共子序列的长度为dp[i][j]
  if(nums[i -1 ] = nums[j - 1]){
      dp[i][j] = dp[i-1][j-1] + 1;
  }else{
      dp[i][j] = Math.max(dp[i][j-1],dp[i-1][j]);
  }
      
  ~~~

- 遍历顺序：因为这道题是从左上和上推出下，所以遍历顺序：从左到右，从上到下

~~~JAVA
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        char[] nums1 = text1.toCharArray();
        char[] nums2 = text2.toCharArray();
        
        int n1 = nums1.length;
        int n2 = nums2.length;
        // dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的最长重复子数组的长度
        int[][] dp = new int[n1 + 1][n2 + 1];
        // 初始化：由于第一行第一列都没有意义，所以全都初始为0
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else{
                    dp[i][j] = Math.max(dp[i-1][j] , dp[i][j-1]);
                }
            }
        }
        int ans = 0;
        for(int[] arr : dp){
            for(int ar : arr){
                ans = Math.max(ar, ans);
            }
        }
        return ans;
    }
}
~~~

### 1035.不相交的线

![image-20240304092856756](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304092856756.png)

<font color=#00f>**和上一题一模一样**</font>

~~~java
class Solution {
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        // dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的最长重复子数组的长度
        int[][] dp = new int[n1 + 1][n2 + 1];
        // 初始化：由于第一行第一列都没有意义，所以全都初始为0
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else{
                    dp[i][j] = Math.max(dp[i-1][j] , dp[i][j-1]);
                }
            }
        }
        int ans = 0;
        for(int[] arr : dp){
            for(int ar : arr){
                ans = Math.max(ar, ans);
            }
        }
        return ans;
    }
}
~~~

### 53.最大子数组和

![image-20240304093055241](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304093055241.png)

- ~~~java
  //dp数组的含义，以i为结尾子数组的最大和
  dp[i] = Math.max(dp[i-1] + nums[i], nums[i]);
  //初始化：因为dp[0] = nums[0]
  ~~~

~~~java
class Solution {
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[0] = nums[0];
        for(int i = 1; i < n ; i++){
            dp[i] = Math.max(dp[i-1] + nums[i], nums[i]);
        }
        int ans = Integer.MIN_VALUE;
        for(int d : dp){
            ans = Math.max(d, ans);
        }
        return ans;
    }
}
~~~

### 392.判断子序列

![image-20240304095152217](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304095152217.png)

<font color=#00f>**这道题用双指针很简单就能做，但是为了特意联系dp方法，所以使用动态规划做**</font>

~~~java
class Solution {
    public boolean isSubsequence(String s, String t) {
        char[] nums1 = s.toCharArray();
        char[] nums2 = t.toCharArray();

        int n1 = nums1.length;
        int n2 = nums2.length;

        int[][] dp = new int[n1 + 1][n2 + 1];
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[n1][n2] == n1;
    }
}
~~~

### ==115.==不同的子序列

![image-20240304103535834](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304103535834.png)

~~~java
//dp[i][j]:以i-1为结尾的s中有以j-1为结尾的个数为dp[i][j];
if(s[i-1] == t[j-1]){
    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]//这种情况是使用了s[i-1]的情况，和没有使用s[i-1]的情况
}else{
    dp[i][j] = dp[i-1][j];
}
//初始化第一行和第一列，推导过程是从左上角和正上方往下推导
//第一行代表s不为空，t为空，方法只有一种，将s中所有元素都删除所以第一行初始为1
//第一列代表s为空，t不为空，所以方法有0种
~~~

~~~java
class Solution {
    public int numDistinct(String s, String t) {
        char[] nums1 = s.toCharArray();
        char[] nums2 = t.toCharArray();

        int n1 = nums1.length;
        int n2 = nums2.length;
        // dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的最长重复子数组的长度
        int[][] dp = new int[n1 + 1][n2 + 1];
        // 初始化第一行和第一列，推导过程是从左上角和正上方往下推导
        // 第一列代表s不为空，t为空，方法只有一种，将s中所有元素都删除所以第一行初始为1
        // 第一行代表s为空，t不为空，所以方法有0种
        for(int i=0; i< n1 + 1; i++){
            dp[i][0] = 1;
        }
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if(nums1[i-1] == nums2[j-1]){
                    //这种情况是使用了s[i-1]的情况，和没有使用s[i-1]的情况
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j];
                }else{
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[n1][n2];
    }
}
~~~

### ==583.==两个字符串的删除操作

![image-20240304105539136](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304105539136.png)

> [!NOTE]
>
> ~~~java
> //dp[i] [j] : 以i-1为结尾word1字段和以j-1为结尾的word2字段相同最小操作次数为dp[i][j]
> if(word2[i-1] == word2[j-1]){
>     dp[i][j] = dp[i-1][j-1];
> }else{
>     //当字符不相同时:
>     //需要删除一个字符时：要么删word1[i-1]要么删word2[j-1]，两个操作取最小的:dp[i][j] =Math.min(dp[i-1][j] + 1, dp[i][j-1] + 1)
>     //需要两个字符都删除时：dp[i-1][j-1] + 2
>     dp[i][j] = Math.min(Math.min(dp[i-1][j] + 1, dp[i][j-1] + 1), dp[i-1][j-1] + 2);
> }
> //初始化：第一行第一列都需要初始化 dp[0][j] = j;dp[i][0] = i;
> ~~~
>
> 



~~~java
class Solution {
    public int minDistance(String word1, String word2) {
        char[] nums1 = word1.toCharArray();
        char[] nums2 = word2.toCharArray();

        int n1 = nums1.length;
        int n2 = nums2.length;
        // dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的最长重复子数组的长度
        int[][] dp = new int[n1 + 1][n2 + 1];
        // /初始化：第一行第一列都需要初始化 dp[0][j] = j;dp[i][0] = i;
        for(int i=0; i< n1 + 1; i++){
            dp[i][0] = i;
        }
        for(int j=0; j< n2 + 1; j++){
            dp[0][j] = j;
        }
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if(nums1[i-1] == nums2[j-1]){
                    dp[i][j] = dp[i-1][j-1];
                }else{
                    dp[i][j] = Math.min(Math.min(dp[i-1][j] + 1, dp[i][j-1] + 1), dp[i-1][j-1] + 2);
                }
            }
        }
        return dp[n1][n2];
    }
}
~~~

### ==72.==编辑距离

![image-20240304141841510](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304141841510.png)

>  [!NOTE]
>
> 删除和添加是一样的，所以只考虑一个就行
>
> ~~~java
> //dp[i] [j] : 以i-1为结尾word1字段和以j-1为结尾的word2字段相同最小操作次数为dp[i][j]
> if(word2[i-1] == word2[j-1]){
>     dp[i][j] = dp[i-1][j-1];
> }else{
>     //当字符不相同时:
>     //需要删除一个字符时：要么删word1[i-1]要么删word2[j-1]，两个操作取最小的:dp[i][j] =Math.min(dp[i-1][j] + 1, dp[i][j-1] + 1)
>     //需要更改一个字符时：dp[i][j] = dp[i-1][j-1] + 1;
>     dp[i][j] = Math.min(Math.min(dp[i-1][j] + 1, dp[i][j-1] + 1), dp[i-1][j-1] + 1);
> }
> //初始化：第一行第一列都需要初始化 dp[0][j] = j;dp[i][0] = i;
> ~~~
>
> 

~~~java
class Solution {
    public int minDistance(String word1, String word2) {
        char[] nums1 = word1.toCharArray();
        char[] nums2 = word2.toCharArray();

        int n1 = nums1.length;
        int n2 = nums2.length;
        // dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的最长重复子数组的长度
        int[][] dp = new int[n1 + 1][n2 + 1];
        // /初始化：第一行第一列都需要初始化 dp[0][j] = j;dp[i][0] = i;
        for(int i=0; i< n1 + 1; i++){
            dp[i][0] = i;
        }
        for(int j=0; j< n2 + 1; j++){
            dp[0][j] = j;
        }
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if(nums1[i-1] == nums2[j-1]){
                    dp[i][j] = dp[i-1][j-1];
                }else{
                    dp[i][j] = Math.min(Math.min(dp[i-1][j] + 1, dp[i][j-1] + 1), dp[i-1][j-1] + 1);
                }
            }
        }
        return dp[n1][n2];
    }
}
~~~

### ==647.==回文子串

![image-20240304144134764](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304144134764.png)

> [!NOTE]
>
> ~~~java
> //dp[i][j]含义：[i,j]是回文子串，true
> //递推公式
> if(s[i] == s[j]){
>     if(j-i<=1){
>         dp[i][j] = true;
>         res++;
>     }else if(dp[i+1][j-1] == true){
>         dp[i][j] = true;
>         res++;
>     }
> }
> //dp数组初始化为false
> //遍历顺序：因为递推公式是从左下角往右上角进行遍历，所以遍历顺序应该是从下往上遍历，从左往右遍历
> ~~~



~~~java
class Solution {
    public int countSubstrings(String s) {
        char[] chars = s.toCharArray();
        int len = chars.length;
        boolean[][] dp = new boolean[len][len];
        int result = 0;
        for (int i = len - 1; i >= 0; i--) {
            for (int j = i; j < len; j++) {
                if (chars[i] == chars[j]) {
                    if (j - i <= 1) { // 情况一 和 情况二（当前i，j处于一个节点；当前i，j中间没有别的回文子串）
                        result++;
                        dp[i][j] = true;
                    } else if (dp[i + 1][j - 1]) { //情况三
                        result++;
                        dp[i][j] = true;
                    }
                }
            }
        }
        return result;
    }
}

~~~

~~~java
class Solution {
    public int countSubstrings(String s) {
        char[] schar = s.toCharArray();
        int n = schar.length;
        int res = 0;
        boolean[][] dp = new boolean[n][n];
        for (int i = n-1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (schar[i] == schar[j]) {
                    if (j - i <= 1) {
                        dp[i][j] = true;
                        res++;
                    } else if (dp[i + 1][j - 1] == true) {
                        dp[i][j] = true;
                        res++;
                    }
                }
            }
        }
        return res;
    }
}
~~~

<font color=#00f>**优化空间复杂度的方法**</font>

~~~java
中心扩散法：
class Solution {
    public int countSubstrings(String s) {
        int len, ans = 0;
        if (s == null || (len = s.length()) < 1) return 0;
        //总共有2 * len - 1个中心点
        for (int i = 0; i < 2 * len - 1; i++) {
            //通过遍历每个回文中心，向两边扩散，并判断是否回文字串
            //有两种情况，left == right，right = left + 1，这两种回文中心是不一样的
            int left = i / 2, right = left + i % 2;
            while (left >= 0 && right < len && s.charAt(left) == s.charAt(right)) {
                //如果当前是一个回文串，则记录数量
                ans++;
                left--;
                right++;
            }
        }
        return ans;
    }
}

~~~

### 516.最长回文子序列

![image-20240304155827679](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304155827679.png)

> [!NOTE]
>
> ~~~java
> //dp[i][j]的含义：在[i,j]中的回文子序列的长度为dp[i][j]
> //递推公式
> if(nums[i] ==nums[j]){
>     dp[i][j] = dp[i+1][j-1] + 2;
> }else{
>     dp[i][j] = Math.max(Math.max(dp[i+1][j],dp[i][j-1]),dp[i+1][j-1]);
> }
> //初始化，因为i和j在相同的时候dp公式是无法递推的，所以应该初始化dp[i][j]所以i=j时初始化为1
> for(int i = 0; i < nums.length; i++){
>     dp[i][i] = 1;
> }
> //遍历顺序：从下往上从左往右，由递推公式获得
> ~~~
>
> <font color=#00f>**注意这里有个优化写法，可以去掉dp[i+1] [j-1]因为这种情况已经被前面的两种情况给包含了**</font>

~~~java
class Solution {
    public int longestPalindromeSubseq(String s) {
        char[] nums = s.toCharArray();
        int n = nums.length;
        // dp[i][j]的含义：在[i,j]中的回文子序列的长度为dp[i][j]
        int[][] dp = new int[n][n];
        // 初始化对角线的元素为1
        for (int i = 0; i < nums.length; i++) {
            dp[i][i] = 1;
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (nums[i] == nums[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(Math.max(dp[i + 1][j], dp[i][j - 1]), dp[i + 1][j - 1]);
                }
            }
        }
        return dp[0][n-1];
    }
}
~~~

### ==2684.==矩阵中移动的最大次数

![image-20240316210139308](Java重写：灵茶山艾府——基础算法精讲.assets/image-20240316210139308.png)

<font color=#00f>**递归法**</font>

~~~java
class Solution {
    private int ans;

    public int maxMoves(int[][] grid) {
        for (int i = 0; i < grid.length; i++) {
            dfs(i, 0, grid); // 从第一列的任一单元格出发
        }
        return ans;
    }

    private void dfs(int i, int j, int[][] grid) {
        ans = Math.max(ans, j);
        if (ans == grid[0].length - 1) { // ans 已达到最大值
            return;
        }
        // 向右上/右/右下走一步
        for (int k = Math.max(i - 1, 0); k < Math.min(i + 2, grid.length); k++) {
            if (grid[k][j + 1] > grid[i][j]) {
                dfs(k, j + 1, grid);
            }
        }
        grid[i][j] = 0;//这步是记忆化，但是我没看懂
    }
}

~~~

<font color=#00f>**自己改的记忆化**</font>

~~~java
class Solution {
    private int ans;

    private int[][] cache;

    public int maxMoves(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;
        cache = new int[n][m];
        for(int[] temp: cache){
            Arrays.fill(temp,-1);
        }
        for (int i = 0; i < grid.length; i++) {
            dfs(i, 0, grid); // 从第一列的任一单元格出发
        }
        return ans;
    }

    private void dfs(int i, int j, int[][] grid) {
        ans = Math.max(ans, j);
        if (ans == grid[0].length - 1) { // ans 已达到最大值
            return;
        }
        // 向右上/右/右下走一步
        for (int k = Math.max(i - 1, 0); k < Math.min(i + 2, grid.length); k++) {
            if(cache[k][j+1] != -1){
                continue;
            }
            if (grid[k][j + 1] > grid[i][j]) {
                dfs(k, j + 1, grid);
                cache[k][j+1] = ans;
            }
        }
    }
}
~~~

<font color=#00f>**递推的方法**</font>

~~~java
class Solution {
    public int maxMoves(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for (int[] row : grid) {
            row[0] *= -1; // 入队标记
        }
        for (int j = 0; j < n - 1; j++) {
            boolean ok = false;
            for (int i = 0; i < m; i++) {
                if (grid[i][j] > 0) { // 不在队列中
                    continue;
                }
                for (int k = Math.max(i - 1, 0); k < Math.min(i + 2, m); k++) {
                    if (grid[k][j + 1] > -grid[i][j]) {
                        grid[k][j + 1] *= -1; // 入队标记
                        ok = true;
                    }
                }
            }
            if (!ok) { // 无法再往右走了
                return j;
            }
        }
        return n - 1;
    }
}

~~~





## 十八.0-1背包|完全背包

<font color=#f00 size= 6>**0-1背包**</font>

<font color=#00f>**对于二维dp数组，先遍历背包还是先遍历物品都是可以的**</font>
$$
二维dp[i] [j] = max(dp[i-1] [j] , dp[i-1] [j-weight[i]] + value[i])\\
一维dp[j] = max(dp [j] , dp[j-weight[i]] + value[i]
$$
<font color=#f00>**为什么一维dp数组要倒序遍历，是为了避免每个物品被重复添加，只有在倒序遍历物品才能保证物品只添加一次**</font>

[代码随想录：01背包基础问题]([带你学透01背包问题（滚动数组篇） | 从此对背包问题不再迷茫！_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1BU4y177kY/?spm_id_from=333.788&vd_source=ed58d82293197f90b081de760b29a7f6))

![image-20240228101929487](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240228101929487.png)

### ==494.==目标和

![image-20240229125629895](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240229125629895.png)

<font color=#00f>**题目解析：为什么这道题是零一背包，思想：从所有的数中选择出正数的和为p，那么负数的和为sum-p，因为要求p+(sum-p)=target所以，可以求出p的值，这个p的值就是01背包的容量**</font>

- 递推数组的含义dp[j]：容量为j的背包所能所能装满的方式有dp[j]的方式（这里面重量就是价值，价值就是重量）
- 递推公式：dp[j] =  dp[j-nums[i]] + dp[nums[i]];<font color=#f00 size=5>**这个地方不明白**</font>
- 初始化：全都初始化为0
- 遍历顺序：先遍历物品再遍历容量，容量倒序遍历，避免一个物品多次放入

~~~java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        for (int x : nums) target += x;
        if (target < 0 || target % 2 == 1) return 0;
        target /= 2;

        int[] dp= new int[target + 1];
        dp[0] =1;
        for(int i = 0; i < nums.length; i++){
            for( int j = target; j >= nums[i]; j--){
                dp[j] += dp[j- nums[i]];//不明白为什么是这个递推公式
            }
        }
        return dp[target];
    }
}
~~~



<font color=#f00>**递归搜素+保存计算结果**</font>

~~~java
class Solution {
    private int[] nums;
    private int[][] cache;

    public int findTargetSumWays(int[] nums, int target) {
        for (int x : nums) {
            target += x;
        }
        if (target < 0 || target % 2 == 1) {
            return 0;
        }
        target /= 2;// 这个target就相当于背包的容量
        this.nums = nums;
        int n = nums.length;
        cache = new int[n][target + 1];
        for (int i = 0; i < n; i++) {
            Arrays.fill(cache[i], -1);
        }
        return dfs(n - 1, target);

    }

    private int dfs(int i, int c) {
        if (i < 0) {
            return c == 0 ? 1 : 0;
        }
        if (cache[i][c] != -1)
            return cache[i][c];
        if (c < nums[i]) {
            return cache[i][c] = dfs(i - 1, c);
        }
        return cache[i][c] = dfs(i - 1, c) + dfs(i - 1, c - nums[i]);
    }
}
~~~

<font color=#00f>**递归转递推**</font>

~~~java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        for (int x : nums) target += x;
        if (target < 0 || target % 2 == 1) return 0;
        target /= 2;

        int n = nums.length;
        int[][] f = new int[n + 1][target + 1];
        //主要是边界条件老出问题
        f[0][0] = 1;
        for (int i = 0; i < n; ++i)
            for (int c = 0; c <= target; ++c)
                if (c < nums[i]) f[i + 1][c] = f[i][c];
                else f[i + 1][c] = f[i][c] + f[i][c - nums[i]];
        return f[n][target];
    }
}

~~~

### 474.一和零

![image-20240301140126800](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240301140126800.png)

- dp数组的含义：dp[i] [j] 装满i个0和j个1，最大背dp[i] [j]个物品
- dp公式：dp[i] [j] = max(dp[i-x] [j-y] + 1 , dp[i] [j]);
- 初始化：dp[0] [0] = 0; 非零下标初始化为0；
- 遍历顺序先遍历物品，再遍历背包，背包倒序遍历为了使物品只使用一次

~~~java
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        int leng = strs.length;
        int[][] dp = new int[m + 1][n + 1];
        //遍历物品
        for (int i = 0; i < leng; i++) {
            int[] ans = X_Y(strs[i]);
            //遍历背包，其中背包有两个维度，所以要进行两层循环，先后顺序无所谓
            for (int j = m; j >= ans[0]; j--) {
                for (int k = n; k >= ans[1]; k--) {
                    dp[j][k] = Math.max(dp[j - ans[0]][k - ans[1]] + 1, dp[j][k]);
                }
            }
        }
        return dp[m][n];
    }
    //判断单个字符串中0和1的数量并封装在一个数组中返回
    private int[] X_Y(String str) {
        int[] res = new int[2];
        //注意要想使用增强for循环遍历字符串，需要将字符串先转换为字符数组
        for (char c : str.toCharArray()) {
            if (c == '0') {
                res[0] += 1;
            } else {
                res[1] += 1;
            }
        }
        return res;
    }
}
~~~

<font color=#f00 size= 6>**完全背包**</font>

<font color=#00f>**对于完全背包，先遍历物品还是先遍历容量都可以进行求解；但是在完全背包的变形问题上，不同的物品容量遍历顺序是代表着求组合还是求排列的结果**</font>

<font color=#f00 size=4>**总结：**</font>

<font color=#f00>**完全背包求排列数：**</font>

- <font color=#f00>**完全背包所以对于容量的遍历必须是正序遍历**</font>
- <font color=#f00>**求排列数，对于元素顺序有要求，需要先遍历背包容量，再遍历物品**</font>
  - <font color=#f00>**由于是先遍历背包容量，所以不能从当前物品重量开始遍历，要从0开始**</font>
  - <font color=#f00>**由于背包容量是从0开始进行的遍历，所以在写递推公式的时候要先进行判断，当前背包容量是否能放入第i个物品**</font>
- <font color=#f00>**对于求有多少组等问题，一般dp[0] 都不初识化为0**</font>

<font color=#f00>**完全背包求组合数**</font>

- <font color=#f00>**完全背包所以对于容量的遍历必须是正序遍历**</font>
- <font color=#f00>**求组合数，对于元素的顺序没有要求，所以先遍历物品，后遍历背包容量**</font>
- <font color=#f00>**对于求有多少组等问题，一般dp[0]都不初始化为0；**</font>

![image-20240228154532051](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240228154532051.png)

![image-20240228154544842](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240228154544842.png)

### 322.零钱兑换

![image-20240228154034307](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240228154034307.png)

- dp[j]的含义：装满背包容量为j的背包，最少物品为dp[j]个
- dp[j] = min(dp[j- coins[i]] + 1, dp[j]);<font color=#00f>**解释一下为什么是和dp[j]求最小，因为这道题是组合数，先遍历物品再遍历背包，所以在每个物品遍历的时候都会有一个dp[j]，要求出最小的dp[j]**</font>
- 初始化：dp[0] = 0;非零下标初始化Integer.MAX-VALUE;
- 遍历顺序：因为是求组合数，先遍历物品再遍历背包，因为是完全背包，所以正序遍历背包

<font color=#00f>**代码随想录**</font>

~~~java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0]=0;
        for(int i= 0; i < n;i++){
            for(int j = coins[i]; j<=amount; j++){
                dp[j] = Math.min(dp[j-coins[i]]+1,dp[j]);
            }
        }
        return dp[amount] == Integer.MAX_VALUE/2 ? -1:dp[amount];
    }
}
~~~



<font color=#00f>**记忆化搜素**</font>

~~~java
class Solution {
    private int[] coins;
    private int[][] cache;

    public int coinChange(int[] coins, int amount) {
        this.coins = coins;
        int n = coins.length;
        cache = new int[n][amount + 1];
        for (int i = 0; i < n; i++)
            Arrays.fill(cache[i], -1); // -1 表示没用访问过
        int ans = dfs(n - 1, amount);
        return ans < Integer.MAX_VALUE / 2 ? ans : -1;
    }

    private int dfs(int i, int c) {
        if (i < 0) return c == 0 ? 0 : Integer.MAX_VALUE / 2; // 除 2 是防止下面 + 1 溢出
        if (cache[i][c] != -1) return cache[i][c];
        if (c < coins[i]) return cache[i][c] = dfs(i - 1, c);
        return cache[i][c] = Math.min(dfs(i - 1, c), dfs(i, c - coins[i]) + 1);
    }
}

~~~

<font color=#00f>**递归转递推**</font>

~~~java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[][] f = new int[n + 1][amount + 1];
        Arrays.fill(f[0], Integer.MAX_VALUE / 2); // 除 2 是防止下面 + 1 溢出
        f[0][0] = 0;
        for (int i = 0; i < n; ++i)
            for (int c = 0; c <= amount; ++c)
                if (c < coins[i]) f[i + 1][c] = f[i][c];
                else f[i + 1][c] = Math.min(f[i][c], f[i + 1][c - coins[i]] + 1);
        int ans = f[n][amount];
        return ans < Integer.MAX_VALUE / 2 ? ans : -1;
    }
}

~~~



![image-20240227171529326](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227171529326.png)

### 518.零钱兑换II

![image-20240301150804754](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240301150804754.png)

<font color=#00f>**这道题完全背包求解组合数**</font>

- dp[j]装满背包容量为j的数组，有dp[j]中方法
- dp[j] += dp[j-coins[i]];
- dp[0] = 1;非零下标初始化为1，因为递推公式是累加的
- 遍历顺序：<font color=#f00>**这道题是组合数：所以应该先遍历物品再遍历容量，因为是完全背包所以是从小到大遍历容量**</font>
- <font color=#f00>**如果是排列数应该是先遍历容量，再遍历物品，因为如果先遍历物品，获得的情况永远是先遍历到的物品在后遍历到的物品的前面，无法满足排列数的所有需求**</font>

~~~java
class Solution {
    public int change(int amount, int[] coins) {
        int n = coins.length;
        // dp[j]数组的含义：容量为j的背包有dp[j]种方法可以凑成target大小的值
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int i = 0; i < n; i++) {
            for (int j = coins[i]; j <= amount; j++) {
                dp[j] += dp[j - coins[i]];
            }
        }
        return dp[amount];
    }
}
~~~

### ==377.==组合总和IV

![image-20240301152020897](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240301152020897.png)

- dp[j]含义：容量为j的完全背包，组成target的组合个数
- dp[j] += dp[j-nums[i]];
- dp[0] = 1;对于这种求组合个数往往dp[0] = 1
- 遍历顺序：因为是完全背包求<font color=#f00>**排列数**</font>，所以先遍历背包容量再遍历物品

~~~java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int n = nums.length;
        // dp[j]数组的含义：容量为j的背包有dp[j]种方法可以凑成target大小的值
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int j = 0; j <= target; j++) {
            for (int i = 0; i < n; i++) {
                if(nums[i] <= j) dp[j] += dp[j - nums[i]];
            }
        }
        return dp[target];
    }
}
~~~

~~~java
class Solution {
    int sum = 0;
    int ans = 0;

    public int combinationSum4(int[] nums, int target) {
        int n = nums.length;

    }

    private void dfs(int i) {
        if (i == n - 1) {
            if (sum == target) {
                ans++;
            }
            return;
        }
        for (int i = 0; i < n; i++) {
            dfs(i);
        }
    }
}
~~~



### ==2915.==和为目标值的最长子序列的长度

![image-20240228163958601](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240228163958601.png)

~~~java
class Solution {
    int[] nums;
    int[][] cache;
    public int lengthOfLongestSubsequence(List<Integer> Nums, int target) {
        nums = Nums.stream().mapToInt(i -> i).toArray();
        int res = -1;
        int n = nums.length;
        cache = new int[n][target+1];
        for(int i = 0; i < n; i++)
            Arrays.fill(cache[i], -2);
        return dfs(n-1, target);
    }
    
    public int dfs(int i, int tot){
        if(i < 0)
            return tot == 0 ? 0 : Integer.MIN_VALUE;
        if(cache[i][tot] >= -1) return cache[i][tot];
        //初始化一个记录最大子序列长度的int变量
        int res = -1;
        res = Math.max(res, dfs(i-1, tot));//判断不选第i节点时最大子序列长度与res的值大小
        if(tot - nums[i] >= 0){//当剩余容量可以容纳i元素时
            int tmp = dfs(i-1, tot-nums[i]);//选当前i元素时容量为tot-nums[i]最大元素个数
            if(tmp != -1){
                res = Math.max(res, tmp + 1);
            }
        }
        return cache[i][tot] = res;
    }
}
~~~

<font color=#f00>**转为递推**</font>



~~~java
class Solution {
    public int lengthOfLongestSubsequence(List<Integer> NUMS, int target) {
        int[] nums = NUMS.stream().mapToInt(i -> i).toArray();
        int n = nums.length;
        int[][] f = new int[n+1][target+1];
        Arrays.fill(f[0], Integer.MIN_VALUE);
        f[0][0] = 0;
        for(int i = 0; i < n; i++){
            Arrays.fill(f[i+1], Integer.MIN_VALUE);
            for(int tot = 0; tot <= target; tot++){
                f[i+1][tot] = Math.max(f[i+1][tot], f[i][tot]);
                if(tot - nums[i] >= 0)
                    f[i+1][tot] = Math.max(f[i+1][tot], f[i][tot-nums[i]] + 1);
            }
        }
        return f[n][target] < -1 ? -1 : f[n][target];
    }
}
~~~

<font color=#f00>**空间优化**</font>

~~~java
class Solution {
    public int lengthOfLongestSubsequence(List<Integer> NUMS, int target) {
        int[] nums = NUMS.stream().mapToInt(i -> i).toArray();
        int n = nums.length;
        int[] f = new int[target+1];
        Arrays.fill(f, Integer.MIN_VALUE);
        f[0] = 0;
        for(int i = 0; i < n; i++){
            int[] tmp = Arrays.copyOf(f, f.length);
            Arrays.fill(f, Integer.MIN_VALUE);
            for(int tot = 0; tot <= target; tot++){
                f[tot] = Math.max(f[tot], tmp[tot]);
                if(tot - nums[i] >= 0)
                    f[tot] = Math.max(f[tot], tmp[tot-nums[i]] + 1);
            }
        }
        return f[target] < -1 ? -1 : f[target];
    }
}
~~~

<font color=#f00>**灵神代码**</font>

~~~java
class Solution {
    public int lengthOfLongestSubsequence(List<Integer> nums, int target) {
        int[] f = new int[target + 1];
        Arrays.fill(f, Integer.MIN_VALUE);
        f[0] = 0;
        int s = 0;
        for (int x : nums) {
            s = Math.min(s + x, target);
            for (int j = s; j >= x; j--) {
                f[j] = Math.max(f[j], f[j - x] + 1);
            }
        }
        return f[target] > 0 ? f[target] : -1;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/length-of-the-longest-subsequence-that-sums-to-target/solutions/2502839/mo-ban-qia-hao-zhuang-man-xing-0-1-bei-b-0nca/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



### 416.分割等和子集

<font color=#00f>**用数组中的数，装满容量为sum/2的背包**</font>

- dp[j]：容量为j的背包最大价值为dp[j];
- 状态转移方程：dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
- 初始化：dp[0] = 0; 因为要选取最大值所以选取非零最小值初始化为0；
- 遍历顺序：先遍历物品，之后倒序遍历背包；

![image-20240228171324292](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240228171324292.png)

<font color=#00f>**使用代码随想录中的思想，一维数组，一遍过**</font>

~~~java
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length;

        //「等和子集」的和必然是总和的一半
        int sum = 0;
        for (int i : nums) sum += i;
        int target = sum / 2;
        
        // 对应了总和为奇数的情况，注定不能被分为两个「等和子集」
        if (target * 2 != sum) return false;

        int[] dp= new int[target+1];

        //初始化dp数组为0，因为默认就是0所以不用进行初始化
        for(int i = 0; i < n ; i++){
            for(int j = target; j >= nums[i]; j--){//只要这里使用了倒序遍历就能保证每个物品只能选一次
                dp[j] = Math.max(dp[j], dp[j-nums[i]] + nums[i]);
            }
        }
        return dp[target] == target ;
    }
}
~~~

<font color=#00f>**第一次自己写的代码，没有通过**</font>

~~~java
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        int n = nums.length;
        int target = sum / 2;
        if (sum % 2 == 1 || n == 1) {
            return false;
        }
        int[][] f = new int[n + 1][target + 1];
        f[0][0] = 0;
        for (int i = 0; i < n; i++) {
            Arrays.fill(f[i+1],0);
            for (int c = 0; c <= target; c++) {
                if (nums[i] > c) {
                    f[i + 1][c] = f[i][c];
                }else{
                    f[i + 1][c] = f[i][c] + f[i][c - nums[i]];
                }
            }
        }
        return f[n][target] == 0 ? false : true;
    }
}
~~~

<font color=#f00>**宫水三叶题解：思路和自己的思路完全一样，就是代码没有成功实现，自己存在一点逻辑错误，就是不能将dp的选和不选的情况进行相加**</font>

~~~java
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length;

        //「等和子集」的和必然是总和的一半
        int sum = 0;
        for (int i : nums) sum += i;
        int target = sum / 2;
        
        // 对应了总和为奇数的情况，注定不能被分为两个「等和子集」
        if (target * 2 != sum) return false;

        int[][] f = new int[n][target + 1];
        // 先处理考虑第 1 件物品的情况
        for (int j = 0; j <= target; j++) {
            f[0][j] = j >= nums[0] ? nums[0] : 0;
        }

        // 再处理考虑其余物品的情况
        for (int i = 1; i < n; i++) {
            int t = nums[i];
            for (int j = 0; j <= target; j++) {
                // 不选第 i 件物品
                int no = f[i-1][j];
                // 选第 i 件物品
                int yes = j >= t ? f[i-1][j-t] + t : 0;
                f[i][j] = Math.max(no, yes);
            }
        }
        // 如果最大价值等于 target，说明可以拆分成两个「等和子集」
        return f[n-1][target] == target;
    }
}

作者：宫水三叶
链接：https://leetcode.cn/problems/partition-equal-subset-sum/solutions/693903/gong-shui-san-xie-bei-bao-wen-ti-shang-r-ln14/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

### 1049.最后一块石头的重量II

![image-20240301122706943](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240301122706943.png)

<font color=#00f>**题解：这道题的思想还是尽可能的将一堆石头分成重量最接近的两堆石头，可以按照装满一个背包容量为总重量一般的01背包进行计算**</font>

- dp[j]含义：装满一个容量为j的背包的最大值（其中value也是重量）。<font color=#00f>**解释一下：因为已知石头的重量和石头的价值是一样的（这个自己规定的）背包容量是固定的，所以求石头的最大价值，就是尽可能的将背包装满**</font>
- 递推公式：dp[j] = max(dp[j] ,dp[j-stones[i]] + stones[i]);
- 初始化方式：因为要求最大值所以初识化为0，第一行要进行初始化，当容量大于第一个物品的重量的时候需要将其改为第一个物品的重量
- 递推顺序：先遍历物品，再遍历背包，背包要倒序进行遍历

~~~java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int sum = 0;
        for (int i : stones) {
            sum += i;
        }
        int target = sum >> 1;
        //初始化dp数组
        int[] dp = new int[target + 1];
        for (int i = 0; i < stones.length; i++) {
            //采用倒序
            for (int j = target; j >= stones[i]; j--) {
                //两种情况，要么放，要么不放
                dp[j] = Math.max(dp[j], dp[j - stones[i]] + stones[i]);
            }
        }
        return sum - 2 * dp[target];
    }
}

作者：代码随想录
链接：https://leetcode.cn/problems/last-stone-weight-ii/solutions/975313/dai-ma-sui-xiang-lu-bang-ni-ba-0-1bei-ba-mlpm/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

<font color=#00f>**自己写的代码：有点小问题看下面的注解**</font>

~~~java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int n = stones.length;

        // 「等和子集」的和必然是总和的一半
        int sum = 0;
        for (int stone : stones)
            sum += stone;
        int target = sum / 2;
        // dp数组定义：装满一个容量为j的背包的最大值（其中value也是重量）
        int[] dp = new int[target + 1];
        // 初始化第一行，为什么初始化第一行反而出现问题了呢，因为初始化第一行之后相当于第一个物品已经进行了放置，下面的遍历物品应该从i=1开始。
        for (int i = 0; i <= target; i++) {
            if (stones[0] <= i) {
                dp[i] = stones[0];
            }
        }
        for (int i = 1; i < n; i++) {
            for (int j = target; j >= stones[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - stones[i]] + stones[i]);
            }
        }
        System.out.println(dp[target]);
        return sum - dp[target]-dp[target];
    }
}
~~~



### 279.完全平方数

![image-20240229130523816](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240229130523816.png)

- dp[j]:装满容量为j的宝贝最少的物品为dp[j];
- dp[j] = min(dp[j - nums[i]] + 1, dp[j])

~~~java
class Solution {
    // 版本一，先遍历物品, 再遍历背包
    public int numSquares(int n) {
        int max = Integer.MAX_VALUE;
        int[] dp = new int[n + 1];
        //初始化
        for (int j = 0; j <= n; j++) {
            dp[j] = max;
        }
        //当和为0时，组合的个数为0
        dp[0] = 0;
        // 遍历物品
        for (int i = 1; i * i <= n; i++) {
            // 遍历背包
            for (int j = i * i; j <= n; j++) {
                if (dp[j - i * i] != max) {
                    dp[j] = Math.min(dp[j], dp[j - i * i] + 1);
                }
            }
        }
        return dp[n];
    }
}
~~~

### 139.单词拆分

![image-20240302111353936](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240302111353936.png)

- 递推公式：if([j,i]这段字符在字典中 && dp[i] == true) dp[i]= true
- 初始化：dp[0] = true,因为题目说明了字符串长度肯定是大于等于1的，所以完全是为了递推公式

~~~java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        HashSet<String> set = new HashSet<>(wordDict);
        boolean[] valid = new boolean[s.length() + 1];
        valid[0] = true;

        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i && !valid[i]; j++) {
                if (set.contains(s.substring(j, i)) && valid[j]) {
                    valid[i] = true;
                }
            }
        }

        return valid[s.length()];
    }
}

~~~

<font color=#00f>**回溯+记忆化搜素**</font>

~~~java
// 回溯法+记忆化
class Solution {
    private Set<String> set;
    private int[] memo;
    public boolean wordBreak(String s, List<String> wordDict) {
        memo = new int[s.length()];
        set = new HashSet<>(wordDict);
        return backtracking(s, 0);
    }

    public boolean backtracking(String s, int startIndex) {
        // System.out.println(startIndex);
        if (startIndex == s.length()) {
            return true;
        }
        if (memo[startIndex] == -1) {
            return false;
        }

        for (int i = startIndex; i < s.length(); i++) {
            String sub = s.substring(startIndex, i + 1);
	    // 拆分出来的单词无法匹配
            if (!set.contains(sub)) {
                continue;                
            }
            boolean res = backtracking(s, i + 1);
            if (res) return true;
        }
        // 这里是关键，找遍了startIndex~s.length()也没能完全匹配，标记从startIndex开始不能找到
        memo[startIndex] = -1;
        return false;
    }
}

~~~





## 十九.线性DP|最长公共子序列

### 1143.最长公共子序列

![image-20240307103119452](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240307103119452.png)

~~~java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        char[] nums1 = text1.toCharArray();
        char[] nums2 = text2.toCharArray();
        
        int n1 = nums1.length;
        int n2 = nums2.length;
        // dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的最长重复子数组的长度
        int[][] dp = new int[n1 + 1][n2 + 1];
        // 初始化：由于第一行第一列都没有意义，所以全都初始为0
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else{
                    dp[i][j] = Math.max(dp[i-1][j] , dp[i][j-1]);
                }
            }
        }
        int ans = 0;
        for(int[] arr : dp){
            for(int ar : arr){
                ans = Math.max(ar, ans);
            }
        }
        return ans;
    }
}
~~~

### ==72.==编辑距离

![image-20240307103620600](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240307103620600.png)

<font color=#aa0 size=5>**第二遍做：对于增删改的dp传递还是有点不熟悉，没有一次做出来**</font>

~~~java
class Solution {
    public int minDistance(String word1, String word2) {
        char[] nums1 = word1.toCharArray();
        char[] nums2 = word2.toCharArray();

        int n1 = nums1.length;
        int n2 = nums2.length;
        // dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的最长重复子数组的长度
        int[][] dp = new int[n1 + 1][n2 + 1];
        //初始化：第一行第一列都需要初始化 dp[0][j] = j;dp[i][0] = i;
        for(int i=0; i< n1 + 1; i++){
            dp[i][0] = i;
        }
        for(int j=0; j< n2 + 1; j++){
            dp[0][j] = j;
        }
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if(nums1[i-1] == nums2[j-1]){
                    dp[i][j] = dp[i-1][j-1];
                }else{
                    dp[i][j] = Math.min(Math.min(dp[i-1][j] + 1, dp[i][j-1] + 1), dp[i-1][j-1] + 1);
                }
            }
        }
        return dp[n1][n2];
    }
}
~~~



![image-20240227171649170](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227171649170.png)

### 583.两个字符串的删除操作

![image-20240307104318095](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240307104318095.png)

~~~java
class Solution {
    public int minDistance(String word1, String word2) {
        char[] nums1 = word1.toCharArray();
        char[] nums2 = word2.toCharArray();

        int n1 = nums1.length;
        int n2 = nums2.length;
        // dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的最长重复子数组的长度
        int[][] dp = new int[n1 + 1][n2 + 1];
        // /初始化：第一行第一列都需要初始化 dp[0][j] = j;dp[i][0] = i;
        for(int i=0; i< n1 + 1; i++){
            dp[i][0] = i;
        }
        for(int j=0; j< n2 + 1; j++){
            dp[0][j] = j;
        }
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if(nums1[i-1] == nums2[j-1]){
                    dp[i][j] = dp[i-1][j-1];
                }else{
                    dp[i][j] = Math.min(Math.min(dp[i-1][j] + 1, dp[i][j-1] + 1), dp[i-1][j-1] + 2);
                }
            }
        }
        return dp[n1][n2];
    }
}
~~~

### 712.两个字符串的最小ACSII删除和

![image-20240307104455449](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240307104455449.png)

<font color=#00f>**自己写的代码，主要注意点是char类型直接转化为int类型，转化出的就是对应的ASCII码**</font>

~~~java
class Solution {
    public int minimumDeleteSum(String s1, String s2) {
        char[] nums1 = s1.toCharArray();
        char[] nums2 = s2.toCharArray();

        int n1 = nums1.length;
        int n2 = nums2.length;
        // dp[i] [j]:以i-1为结尾的nums1，以j-1为结尾的nums2的两个元素相同删除的ascii码的最小值
        int[][] dp = new int[n1 + 1][n2 + 1];
        // /初始化：第一行第一列都需要初始化 dp[0][j] = j;dp[i][0] = i;
        dp[0][0] = 0;
        for (int i = 1; i < n1 + 1; i++) {
            dp[i][0] = (dp[i - 1][0] + ((int)nums1[i - 1]));
        }
        for (int j = 1; j < n2 + 1; j++) {
            dp[0][j] = (dp[0][j - 1] + ((int)nums2[j - 1]));
        }
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j] + ((int)nums1[i -1]), dp[i][j - 1] + ((int)nums2[j -1]));
                }
            }
        }
        return dp[n1][n2];
    }
}
~~~



~~~java
class Solution {
    public int minimumDeleteSum(String s1, String s2) {
        int m = s1.length(), n = s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = dp[i - 1][0] + s1.codePointAt(i - 1);
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = dp[0][j - 1] + s2.codePointAt(j - 1);
        }
        for (int i = 1; i <= m; i++) {
            int code1 = s1.codePointAt(i - 1);
            for (int j = 1; j <= n; j++) {
                int code2 = s2.codePointAt(j - 1);
                if (code1 == code2) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j] + code1, dp[i][j - 1] + code2);
                }
            }
        }
        return dp[m][n];
    }
}
~~~



### ==1458.==两个子序列的最大点积

![image-20240307110324367](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240307110324367.png)

~~~java
class Solution {
    public int maxDotProduct(int[] nums1, int[] nums2) {
        int n1 = nums1.length,n2 = nums2.length;
        //dp[i+1][j+1]表示分别在nums1[0,i]和nums2[0,j]的范围内找到的子序列乘积的最大值
        int[][] dp = new int[n1+1][n2+1];
        //索引0的位置无意义，值为负无穷
        Arrays.fill(dp[0], -0x3f3f3f3f);
        for (int i = 0; i <= n1; i++) {
            dp[i][0] = -0x3f3f3f3f;
        }
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                int mul = nums1[i]*nums2[j];
                //对于每个dp[i+1][j+1]，选择是否累加当前乘积
                //如果不选，则从上一位(dp[i][j+1])或左一位(dp[i+1][j])转移最大值
                //如果选，则再分2种情况：
                //1.不利用之前的状态，从零开始计算，此时值为乘积
                //2.利用之前的状态，此时值为乘积+dp[i][j]
                dp[i+1][j+1] = Math.max(Math.max(dp[i+1][j],dp[i][j+1]),mul+Math.max(dp[i][j],0));
            }
        }
        return dp[n1][n2];
    }
}

~~~



<font color=#f00>**别人的题解：记忆化搜素**</font>

~~~java
class Solution {
    private int l1, l2;
    private int[] nums1, nums2;
    private int[][] cache;

    public int maxDotProduct(int[] nums1, int[] nums2) {
        l1 = nums1.length;
        l2 = nums2.length;
        this.nums1 = nums1;
        this.nums2 = nums2;
        cache = new int[l1][l2];
        for (int i = 0; i < l1; i++) Arrays.fill(cache[i], -1);

        int ans = dfs(l1 -1, l2 - 1);

        if (ans == 0) {
            int min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
            for (int x : nums1) if (Math.abs(x) < Math.abs(min1)) min1 = x;
            for (int x : nums2) if (Math.abs(x) < Math.abs(min2)) min2 = x;
            return min1 * min2;
        }
        return ans;
    }

    private int dfs(int i, int j) {
        if (i < 0 || j < 0) return 0;
        if (cache[i][j] != -1) return cache[i][j]; 
        return cache[i][j] = Math.max(Math.max(dfs(i - 1, j), dfs(i, j - 1)), dfs(i - 1, j - 1) + nums1[i] * nums2[j]); 
    }
}
~~~



### 97.交错字符串

![image-20240307110346057](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240307110346057.png)

~~~java
class Solution {
    public boolean isInterleave(String s1, String s2, String s3) {
        char[] num1 = s1.toCharArray();
        char[] num2 = s2.toCharArray();
        char[] num3 = s3.toCharArray();
        if(num1.length + num2.length != num3.length){
            return false;
        }
        //dp[i][j]数组的含义，nums1的前i-1项,和nums的前j-1项可以组成字符串3
        boolean[][] dp = new boolean[num1.length+1][num2.length+1];
        //初始化数组
        dp[0][0] = true;
        for(int i=1; i<= num1.length; i++){
            if(dp[i-1][0] && num1[i - 1] == num3[i - 1]){
                dp[i][0] = true;
            }else{
                dp[i][0] = false;
            }
        }
        for(int j=1; j<= num2.length; j++){
            if(dp[0][j-1] && num2[j - 1] == num3[j - 1]){
                dp[0][j] = true;
            }else{
                dp[0][j] = false;
            }
        }
        //进行递推
        for(int i = 1; i<=num1.length; i++){
            for(int j= 1;j <= num2.length;j++){
                if(dp[i-1][j] && num1[i-1] == num3[i+j-1]){
                    dp[i][j] = true;
                }else if(dp[i][j-1] && num2[j - 1] == num3[i+j-1]){
                    dp[i][j] = true;
                }else{
                    dp[i][j] = false;
                }
            }
        }
        return dp[num1.length][num2.length];
    }
}
~~~



## 二十.线性DP|最长递增子序列

### 300.最长递增子序列

![image-20240304174656802](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304174656802.png)

~~~java
class Solution {
    public int lengthOfLIS(int[] nums) {
        //dp[i]以nums[i]为结尾的最长严格递增子序列的个数
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp,1);
        for(int i = 1; i < nums.length; i++){
            for(int j = i -1 ; j >=0; j--){
                if(nums[j] <nums[i]){
                    dp[i] = Math.max(dp[i],dp[j] + 1);
                }
            }
        }
        int res = 0;
        for(int i:dp){
            res = Math.max(i,res);
        }
        return res;
    }
}
~~~



![image-20240227171927239](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240227171927239.png)

### ==673.==最长==递增==子序==列的==个数

<font color=#f00>**怎么都想不特别明白**</font>

![image-20240304174715997](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240304174715997.png)

<font color=#00f>**这道题和上面的那道题还有点不一样，如果单纯的记录每个位置的最长子序列的长度，最后进行遍历统计，会出现因为最后一个值在同一个位置而漏解的情况**</font>

<font color=#f00>**这道题是第300题的升级**</font>

~~~java
class Solution {
    public int findNumberOfLIS(int[] nums) {
        if (nums.length <= 1) return nums.length;
        int[] dp = new int[nums.length];
        for(int i = 0; i < dp.length; i++) dp[i] = 1;
        int[] count = new int[nums.length];
        for(int i = 0; i < count.length; i++) count[i] = 1;

        int maxCount = 0;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    //因为在nums[i]>nums[j]的情况下dp[i]应该是等于dp[j]+1,如果不等于这个结果，证明dp[i]还没被更新
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];
                        //如果已经等于这个结果了，那说明存在比j小的m索引处，使dp[m]==dp[j]，那么此时子序列的个数就是累加
                    } else if (dp[j] + 1 == dp[i]) {
                        count[i] += count[j];
                    }
                }
                if (dp[i] > maxCount) maxCount = dp[i];
            }
        }
        int result = 0;
        for (int i = 0; i < nums.length; i++) {
            if (maxCount == dp[i]) result += count[i];
        }
        return result;
    }
}

~~~



1964.找出到每个位置为止最长的有效障碍赛跑路线

![image-20240308172903659](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240308172903659.png)

~~~java
~~~





1671.得到山形数组的最少删除次数

![image-20240308172922515](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240308172922515.png)



### ==354.==俄罗斯套娃信封问题

![image-20240308173003076](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240308173003076.png)

~~~java
class Solution {
    public int maxEnvelopes(int[][] es) {
        int n = es.length;
        if (n == 0) return n;
        // 因为我们在找第 i 件物品的前一件物品时，会对前面的 i - 1 件物品都遍历一遍，因此第二维（高度）排序与否都不影响
        Arrays.sort(es, (a, b)->a[0]-b[0]);
        int[] f = new int[n]; // f(i) 为考虑前 i 个物品，并以第 i 个物品为结尾的最大值
        int ans = 1;
        for (int i = 0; i < n; i++) {
            // 对于每个 f[i] 都满足最小值为 1
            f[i] = 1; 
            // 枚举第 i 件物品的前一件物品，
            for (int j = i - 1; j >= 0; j--) {
                // 只要有满足条件的前一件物品，我们就尝试使用 f[j] + 1 更新 f[i]
                if (check(es, j, i)) {
                    f[i] = Math.max(f[i], f[j] + 1);
                }
            }
            // 在所有的 f[i] 中取 max 作为 ans
            ans = Math.max(ans, f[i]);
        }
        return ans;
    }
    boolean check(int[][] es, int mid, int i) {
        return es[mid][0] < es[i][0] && es[mid][1] < es[i][1];
    }
}

作者：宫水三叶
链接：https://leetcode.cn/problems/russian-doll-envelopes/solutions/633912/zui-chang-shang-sheng-zi-xu-lie-bian-xin-6s8d/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

<font color=#f00>**二分法**</font>

~~~java
class Solution {
    public int maxEnvelopes(int[][] es) {
        int n = es.length;
        if (n == 0) return n;
        // 由于我们使用了 g 记录高度，因此这里只需将 w 从小到达排序即可
        Arrays.sort(es, (a, b)->a[0] - b[0]);
        // f(i) 为考虑前 i 个物品，并以第 i 个物品为结尾的最大值
        int[] f = new int[n]; 
        // g(i) 记录的是长度为 i 的最长上升子序列的最小「信封高度」
        int[] g = new int[n]; 
        // 因为要取 min，用一个足够大（不可能）的高度初始化
        Arrays.fill(g, Integer.MAX_VALUE); 
        g[0] = 0;
        int ans = 1;
        for (int i = 0, j = 0, len = 1; i < n; i++) {
            // 对于 w 相同的数据，不更新 g 数组
            if (es[i][0] != es[j][0]) {
                // 限制 j 不能越过 i，确保 g 数组中只会出现第 i 个信封前的「历史信封」
                while (j < i) {
                    int prev = f[j], cur = es[j][1];
                    if (prev == len) {
                        // 与当前长度一致了，说明上升序列多增加一位
                        g[len++] = cur;
                    } else {
                        // 始终保留最小的「信封高度」，这样可以确保有更多的信封可以与其行程上升序列
                        // 举例：同样是上升长度为 5 的序列，保留最小高度为 5 记录（而不是保留任意的，比如 10），这样之后高度为 7 8 9 的信封都能形成序列；
                        g[prev] = Math.min(g[prev], cur);
                    }
                    j++;
                }
            }

            // 二分过程
            // g[i] 代表的是上升子序列长度为 i 的「最小信封高度」
            int l = 0, r = len;
            while (l < r) {
                int mid = l + r >> 1;
                // 令 check 条件为 es[i][1] <= g[mid]（代表 w 和 h 都严格小于当前信封）
                // 这样我们找到的就是满足条件，最靠近数组中心点的数据（也就是满足 check 条件的最大下标）
                // 对应回 g[] 数组的含义，其实就是找到 w 和 h 都满足条件的最大上升长度
                if (es[i][1] <= g[mid]) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            // 更新 f[i] 与答案
            f[i] = r;
            ans = Math.max(ans, f[i]);
        }
        return ans;
    }
}

作者：宫水三叶
链接：https://leetcode.cn/problems/russian-doll-envelopes/solutions/633912/zui-chang-shang-sheng-zi-xu-lie-bian-xin-6s8d/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



### 1626.无矛盾的最佳球队

![image-20240308173024812](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240308173024812.png)

<font color=#00f>**最简单的做法，两个数组按照分数排序，之后从后部进行保留**</font>

~~~java
class Solution {
    public int bestTeamScore(int[] scores, int[] ages) {
        int n = scores.length, ans = 0;
        var ids = new Integer[n];
        for (int i = 0; i < n; ++i)
            ids[i] = i;
        Arrays.sort(ids, (i, j) -> scores[i] != scores[j] ? scores[i] - scores[j] : ages[i] - ages[j]);

        var f = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j)
                if (ages[ids[j]] <= ages[ids[i]])
                    f[i] = Math.max(f[i], f[j]);
            f[i] += scores[ids[i]];
            ans = Math.max(ans, f[i]);
        }
        return ans;
    }
}

~~~



## 二十一.状态机DP|买卖股票系列

### 1911.最大子序列交替和

![image-20240308214242251](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240308214242251.png)

~~~java
class Solution {
    public long maxAlternatingSum(int[] nums) {
        int n = nums.length;
        long[][] dp = new long[n][2];
        dp[0][0] = 0;
        dp[0][1] = nums[0];
        for (int i = 1; i < n; i++) {
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] - nums[i]);
            dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] + nums[i]);
        }
        return dp[n-1][1];
    }
}
~~~



## 二十二.区间DP

<font color=#f00 size=5>**区间dp和概率dp都是竞赛中的难题，所以大厂算法题一般不会到这个难度**</font>

### 516.最长回文子系列

![image-20240312094046222](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240312094046222.png)

~~~java
class Solution {
    public int longestPalindromeSubseq(String s) {
        char[] nums = s.toCharArray();
        int n = nums.length;
        // dp[i][j]的含义：在[i,j]中的回文子序列的长度为dp[i][j]
        int[][] dp = new int[n][n];
        // 初始化对角线的元素为1
        for (int i = 0; i < nums.length; i++) {
            dp[i][i] = 1;
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (nums[i] == nums[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(Math.max(dp[i + 1][j], dp[i][j - 1]), dp[i + 1][j - 1]);
                }
            }
        }
        return dp[0][n-1];
    }
}
~~~



### 1039.最优三角剖分

![image-20240312095353490](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240312095353490.png)

~~~java
class Solution {
    public int minScoreTriangulation(int[] values) {
        // dp[i][j]表示从i到j顺时针转动这部分的点的所组成的三角形的最小值为dp[i][j]
        // 递推公式：
        int n = values.length;
        int[][] dp = new int[n][n];

        for (int i = n - 3; i >= 0; i--) {
            for (int j = i + 2; j < n; j++) {
                int res = Integer.MAX_VALUE;
                for (int k = i + 1; k < j; k++) {
                    res = Math.min(dp[i][k] + dp[k][j] + values[i] * values[j] * values[k], res);
                }
                dp[i][j] = res;
            }
        }
        return dp[0][n - 1];
    }
}
~~~





![image-20240305110938289](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240305110938289.png)

### 375.猜数字大小II

![image-20240312102526257](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240312102526257.png)

- 递归思路

  

<font color=#00f>**记忆化搜素**</font>

~~~java
class Solution {
    int[][] cache;

    public int getMoneyAmount(int n) {
        this.cache = new int[n + 1][n + 1];
        for (int i = 0; i < cache.length; i++) {
            Arrays.fill(cache[i], -1);
        }
        return dfs(1, n);
    }

    // 当数字范围为i，j时需要的最少花费
    private int dfs(int i, int j) {
        if (cache[i][j] != -1) {
            return cache[i][j];
        }
        if (j - i == 1) {
            return cache[i][j] = i;
        }
        if (j == i) {
            return 0;
        }
        int res = Integer.MAX_VALUE;
        for(int k = i+ 1; k<j;k++){
            res = Math.min(res, k + Math.max(dfs(i, k - 1), dfs(k + 1, j)));
        }
        return cache[i][j] = res;
    }
}
~~~

<font color=#00f>**递推**</font>

~~~java
// 方法二：递推
    public int getMoneyAmount(int n) {
        int[][] f = new int[n + 1][n + 1];
        for (int i = n; i > 0; i--) {
            for (int j = i; j <= n; j++) {
                if (i == j) {
                    f[i][j] = 0;
                    continue;
                }
                f[i][j] = Integer.MAX_VALUE;
                for (int k = i; k <= j; k++) {
                    f[i][j] = Math.min(f[i][j], k + Math.max(k > i ? f[i][k - 1] : 0, k < j ? f[k + 1][j] : 0));
                }
            }
            System.out.println(Arrays.toString(f[i]));
        }
        return f[1][n];
    }

~~~



132.分割回文串II

![image-20240312140637755](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240312140637755.png)

1312.让字符串成为回文串的最少插入次数

![image-20240312140743705](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240312140743705.png)



1771.由子序列构造最长回文串的长度

![image-20240312140759282](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240312140759282.png)

1547.切棍子的最小成本

![image-20240312140708634](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240312140708634.png)

1000.合并石头的最低成本

![image-20240312140817185](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240312140817185.png)

## 二十三.树形DP

### 543.二叉树的直径

![image-20240309105624328](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240309105624328.png)

<font color=#00f>**自己写的递归方法，视频中的树桩dp也是这样写的**</font>

~~~java
class Solution {
    int ans = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return ans;
    }

    private int dfs(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int left_height = 0, right_height = 0;
        if (node.left != null) {
            left_height = dfs(node.left);
        }
        if (node.right != null) {
            right_height = dfs(node.right);
        }
        ans = Math.max(ans, left_height + right_height );
        return Math.max(left_height, right_height) + 1;

    }
}
~~~



### 124.二叉树中的最大路径和

![image-20240309110817415](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240309110817415.png)

~~~java
class Solution {
    int ans = 0;
    public int maxPathSum(TreeNode root) {
        dfs(root);
        return ans;
    }

    private int dfs(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int leftsum = dfs(node.left);
        int rightsum = dfs(node.right);
        ans = Math.max(ans, leftsum + rightsum + node.val);
        return Math.max(0,Math.max(leftsum, rightsum) + node.val);
    }
}
~~~



### ==2246.==相邻字符不同的最长路径

![image-20240309112126231](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240309112126231.png)

~~~java
// 定义一个名为Solution的类，用于解决最长路径问题
class Solution {
    // 成员变量：邻接表表示图结构
    List<Integer>[] g;
    // 成员变量：字符串s，存储节点字符信息
    String s;
    // 成员变量：记录当前找到的最长路径长度
    int ans;

    /**
     * 主方法：计算给定父节点数组和字符串s中的最长路径长度
     * @param parent 父节点数组，下标i表示节点i的父节点
     * @param s 字符串，每个字符对应图中一个节点
     * @return 最长路径的长度加1（题目要求格式）
     */
    public int longestPath(int[] parent, String s) {
        this.s = s;
        // 获取节点数量
        var n = parent.length;
        // 初始化邻接表
        g = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        // 构建邻接表，添加子节点到父节点的列表中
        for (var i = 1; i < n; i++) {
            g[parent[i]].add(i);
        }

        // 从根节点开始进行深度优先搜索
        dfs(0);

        // 返回处理后的最长路径长度
        return ans + 1;
    }

    /**
     * 深度优先搜索辅助方法：递归地查找以节点x为起点的满足要求的最长路径长度
     * @param x 当前访问节点的下标
     * @return 以节点x为起点的最长路径长度
     */
    private int dfs(int x) {
        //用来保存当前节点的一条最长子路径的长度，因为递归是找以当前节点为父节点的两条长度相加得到的最长路径，所以要用一个变量保存一个最长的，另一个次长的通过递归过程中进行寻找
        var maxLen = 0; // 初始化最大长度为0

        // 遍历节点x的所有子节点
        for (var y : g[x]) {
            // 计算从y节点出发的路径长度，并加上1（因为包含自身）
            var len = dfs(y) + 1;
            
            // 如果当前节点与子节点对应的字符不相同，则更新最长路径长度
            if (s.charAt(y) != s.charAt(x)) {
                ans = Math.max(ans, maxLen + len);//将当前节点的两条子路径相加得到最长路径，与结果进行比较
                maxLen = Math.max(maxLen, len);//判断当前递归到的路径是最长的还是次长的，如果是最长的需要记录保存
            }
        }
        
        // 返回以节点x为起点的最长路径长度
        return maxLen;
    }
}
~~~



![image-20240305110731547](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240305110731547.png)

### ==687.==最长同值路径

![image-20240309150806560](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240309150806560.png)

<font color=#00f>**自己写的ak**</font>

~~~java
class Solution {
    int ans = 0;

    public int longestUnivaluePath(TreeNode root) {
        dfs(root);
        return ans;
    }

    // 递归一个数组，int[0]表示当前最长路径的元素值，int[1]表示最长路径的长度
    private int[] dfs(TreeNode node) {
        if (node == null) {
            return new int[] {-2000, 0};
        }
        
        int[] leftArray = dfs(node.left);
        int[] rightArray = dfs(node.right);
        int left = 0, right = 0;
        if (node.val == leftArray[0]) {
            left = leftArray[1] + 1;
        }
        if (node.val == rightArray[0]) {
            right = rightArray[1] + 1;
        }
        ans = Math.max(ans, left + right);
        return new int[] { node.val, Math.max(left, right) };
    }
}
~~~

<font color=#f00>**宫水三叶题解**</font>

~~~java
class Solution {
    int max = 0;
    public int longestUnivaluePath(TreeNode root) {
        dfs(root);
        return max;
    }
    int dfs(TreeNode root) {
        if (root == null) return 0;
        int ans = 0, cur = 0, l = dfs(root.left), r = dfs(root.right);
        if (root.left != null && root.left.val == root.val) {
            ans = l + 1; cur += l + 1;
        }
        if (root.right != null && root.right.val == root.val) {
            ans = Math.max(ans, r + 1); cur += r + 1;
        }
        max = Math.max(max, cur);
        return ans;
    }
}
~~~

### ==1617.==统计子树中城市之间最大距离

![image-20240309153435789](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240309153435789.png)

<font color=#f00>**这道题很难用到了图论的知识，没学明白**</font>

~~~java
// 定义一个名为Solution的类，用于解决计算具有给定直径的子图数量问题
class Solution {
    // 初始化邻接表数组，存储每座城市的相邻城市列表
    private List<Integer>[] g;
    
    // 初始化布尔数组inSet，记录当前组合中是否选择了该城市
    private boolean[] inSet;
    
    // 初始化结果数组ans，存储具有不同直径的子图个数
    private int[] ans;

    // 记录城市数量
    private int n;
    
    // 临时变量diameter，用于在DFS过程中跟踪当前子图的直径
    private int diameter;

    /**
     * 公共方法：计算所有具有指定直径的子图的数量
     * @param n 城市数量
     * @param edges 边数组，表示城市之间的连接关系
     * @return ans数组，包含从1到n-1（不包括0）每个直径值对应的子图数量
     */
    public int[] countSubgraphsForEachDiameter(int n, int[][] edges) {
        this.n = n; // 初始化城市数量
        
        // 初始化邻接表，每座城市对应一个空的邻居列表
        g = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());

        // 根据边数组构建无向图
        for (var edge : edges) {
            int x = edge[0] - 1, y = edge[1] - 1; // 将编号从1改为0开始
            g[x].add(y);
            g[y].add(x); // 添加双向边
        }

        // 初始化答案数组
        ans = new int[n - 1];

        // 调用回溯法计算所有子图及其直径
        f(0);

        return ans; // 返回结果数组
    }

    // 回溯法递归函数，尝试选择或不选择城市i加入子图
    private void f(int i) {
        if (i == n) { // 所有城市已遍历完
            // 遍历已选择的城市，对每个子集求其直径
            for (int v = 0; v < n; ++v)
                if (inSet[v]) {
                    vis = new boolean[n]; // 重置访问标记数组
                    diameter = 0; // 重置直径计数器
                    dfs(v); // 深度优先遍历求解子图直径
                    break; // 只需计算并统计一个子图的直径即可
                }
            
            // 如果子图的直径大于0且与vis数组一致，说明此直径有效，更新结果数组
            if (diameter > 0 && Arrays.equals(vis, inSet))
                ++ans[diameter - 1]; // 子图直径从1开始计数，所以使用diameter - 1作为索引

            return; // 继续回溯
        }

        // 不选城市i
        f(i + 1);

        // 选城市i
        inSet[i] = true;
        f(i + 1);
        
        // 撤销选择，恢复现场
        inSet[i] = false;
    }

    // 深度优先搜索算法，用于求解树形结构的直径
    private int dfs(int x) {
        vis[x] = true; // 标记节点x已被访问
        
        int maxLen = 0; // 初始化最大路径长度
        for (int y : g[x]) {
            if (!vis[y] && inSet[y]) { // 探索未访问过的、并且被选择的城市y
                int ml = dfs(y) + 1; // 计算到城市y的路径长度，并加1
                diameter = Math.max(diameter, maxLen + ml); // 更新当前子图的直径，取最长路径长度
                maxLen = Math.max(maxLen, ml); // 更新从城市x出发的最大路径长度
            }
        }

        return maxLen; // 返回从城市x出发的最长路径长度
    }
}
~~~

~~~java
class Solution {
    private List<Integer>[] g;
    private boolean[] inSet, vis;
    private int[] ans;
    private int n, diameter;

    public int[] countSubgraphsForEachDiameter(int n, int[][] edges) {
        this.n = n;
        g = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for (var e : edges) {
            int x = e[0] - 1, y = e[1] - 1; // 编号改为从 0 开始
            g[x].add(y);
            g[y].add(x); // 建树
        }

        ans = new int[n - 1];
        inSet = new boolean[n];
        f(0);
        return ans;
    }

    private void f(int i) {
        if (i == n) {
            for (int v = 0; v < n; ++v)
                if (inSet[v]) {
                    vis = new boolean[n];
                    diameter = 0;
                    dfs(v);
                    break;
                }
            if (diameter > 0 && Arrays.equals(vis, inSet))
                ++ans[diameter - 1];
            return;
        }

        // 不选城市 i
        f(i + 1);

        // 选城市 i
        inSet[i] = true;
        f(i + 1);
        inSet[i] = false; // 恢复现场
    }

    // 求树的直径
    private int dfs(int x) {
        vis[x] = true;
        int maxLen = 0;
        for (int y : g[x])
            if (!vis[y] && inSet[y]) {
                int ml = dfs(y) + 1;
                diameter = Math.max(diameter, maxLen + ml);
                maxLen = Math.max(maxLen, ml);
            }
        return maxLen;
    }
}

~~~



### ==2538.==最大价值与最小价值和的差值（反复观看）

![image-20240309153518168](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240309153518168.png)

~~~java
class Solution {
    private List<Integer>[] g;
    private int[] price;
    private long ans;

    public long maxOutput(int n, int[][] edges, int[] price) {
        this.price = price;
        g = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for (var e : edges) {
            int x = e[0], y = e[1];
            g[x].add(y);
            g[y].add(x); // 建树
        }
        dfs(0, -1);
        return ans;
    }

    private long[] dfs(int x, int fa) {
        long p = price[x], maxS1 = p, maxS2 = 0;
        for (var y : g[x])
            if (y != fa) {
                var res = dfs(y, x);
                long s1 = res[0], s2 = res[1];
                // 前面最大带叶子的路径和 + 当前不带叶子的路径和
                // 前面最大不带叶子的路径和 + 当前带叶子的路径和
                ans = Math.max(ans, Math.max(maxS1 + s2, maxS2 + s1));
                maxS1 = Math.max(maxS1, s1 + p);
                maxS2 = Math.max(maxS2, s2 + p); // 这里加上 p 是因为 x 必然不是叶子
            }
        return new long[]{maxS1, maxS2};
    }
}

~~~



## 二十四.树形DP-打家劫舍III

### 337.打家劫舍III

![image-20240303110819319](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240303110819319.png)

<font color=#00f>**记忆化搜素**</font>

~~~java
// 2.递归去偷，记录状态
    // 执行用时：3 ms , 在所有 Java 提交中击败了 56.24% 的用户
    public int rob1(TreeNode root) {
        Map<TreeNode, Integer> memo = new HashMap<>();
        return robAction(root, memo);
    }

    int robAction(TreeNode root, Map<TreeNode, Integer> memo) {
        if (root == null)
            return 0;
        if (memo.containsKey(root))
            return memo.get(root);
        int money = root.val;
        if (root.left != null) {
            money += robAction(root.left.left, memo) + robAction(root.left.right, memo);
        }
        if (root.right != null) {
            money += robAction(root.right.left, memo) + robAction(root.right.right, memo);
        }
        int res = Math.max(money, robAction(root.left, memo) + robAction(root.right, memo));
        memo.put(root, res);
        return res;
    }


~~~

<font color=#00f>**树形dp**</font>

~~~java
// 3.状态标记递归
    // 执行用时：0 ms , 在所有 Java 提交中击败了 100% 的用户
    // 不偷：Max(左孩子不偷，左孩子偷) + Max(右孩子不偷，右孩子偷)
    // root[0] = Math.max(rob(root.left)[0], rob(root.left)[1]) +
    // Math.max(rob(root.right)[0], rob(root.right)[1])
    // 偷：左孩子不偷+ 右孩子不偷 + 当前节点偷
    // root[1] = rob(root.left)[0] + rob(root.right)[0] + root.val;
    public int rob3(TreeNode root) {
        int[] res = robAction1(root);
        return Math.max(res[0], res[1]);
    }

    int[] robAction1(TreeNode root) {
        int res[] = new int[2];
        if (root == null)
            return res;

        int[] left = robAction1(root.left);
        int[] right = robAction1(root.right);

        res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        res[1] = root.val + left[0] + right[0];
        return res;
    }

~~~

<font color=#00f>**自己写的树形dp**</font>

~~~java
class Solution {
    public int rob(TreeNode root) {
        int[] res = dfs(root);
        return Math.max(res[0], res[1]);
    }

    // dp数组是一个一维两个容量的数组，
    // dp[0]代表不偷当前节点所能获得的最大值，
    // dp[1]代表偷当前节点所能获得的最大值
    private int[] dfs(TreeNode node) {
        if (node == null) {
            return new int[] { 0, 0 };
        }
        int[] left_dp = dfs(node.left);
        int[] right_dp = dfs(node.right);
        // 偷当前节点的值
        int val_0 = Math.max(left_dp[0], left_dp[1]) + Math.max(right_dp[0], right_dp[1]);
        // 不偷当前节点的值
        int val_1 = left_dp[0] + right_dp[0] + node.val;
        return new int[] { val_0, val_1 };
    }
}
~~~

![image-20240305111115215](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240305111115215.png)

1377.T秒后青蛙的位置

2646.最小化旅行的价格总和

## 二十五.树形DP-监控二叉树

### ==968.==监控二叉树

![image-20240309153614400](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240309153614400.png)

<font color=#00f>**蓝色：安装摄像头**</font>

<font color=#aa3>**黄色：不安装摄像头，且他的父节点安装摄像头**</font>

<font color=#f00>**红色：不安装摄像头，且它的至少一个儿子安装摄像头**</font>

![image-20240309154326448](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240309154326448.png)

~~~java
class Solution {
    public int minCameraCover(TreeNode root) {
        int[] res = dfs(root);
        return Math.min(res[0], res[2]);
    }

    private int[] dfs(TreeNode node) {
        if (node == null) {
            return new int[]{Integer.MAX_VALUE / 2, 0, 0}; // 除 2 防止加法溢出
        }
        int[] left = dfs(node.left);
        int[] right = dfs(node.right);
        int choose = Math.min(left[0], left[1]) + Math.min(right[0], right[1]) + 1;
        int byFa = Math.min(left[0], left[2]) + Math.min(right[0], right[2]);
        int byChildren = Math.min(Math.min(left[0] + right[2], left[2] + right[0]), left[0] + right[0]);
        return new int[]{choose, byFa, byChildren};
    }
}

~~~



![image-20240305111844037](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240305111844037.png)

保安站岗

### ==LCP34.==二叉树染色

![image-20240309160010747](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240309160010747.png)

~~~java
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
    public int maxValue(TreeNode root, int k) {
        //dp数组含义当前节点为第i个连续染色的节点时，最大价值和是多少
        int[] dp = getDP(root, k);
        int ans = Integer.MIN_VALUE;
        for(int i = 0; i <= k; i ++){
            ans = Math.max(ans, dp[i]);
        }
        return ans;
    }
    public int[] getDP(TreeNode root, int maxCount){
        // 当前节点为空时直接返回全为0的数组
        int[] dp = new int[maxCount + 1];
        if(root == null) return dp;

        // 获取左右子节点的dp结果
        int[] ldp = getDP(root.left, maxCount);
        int[] rdp = getDP(root.right, maxCount);

        // 当前节点不染色，最大值为左右子树染色最大值的和
        int lMax = Integer.MIN_VALUE;
        int rMax = Integer.MIN_VALUE;
        for(int i = 0; i <= maxCount; i ++){
            lMax = Math.max(lMax, ldp[i]);
            rMax = Math.max(rMax, rdp[i]);
        }
        dp[0] = lMax + rMax;

        // 当前节点染色个数为i时，取左右子节点染色个数和为(i-1)的所有情况的最大值
        for(int i = 1; i <= maxCount; i ++){
            for(int j = 0; j < i; j ++){
                dp[i] = Math.max(dp[i], root.val + ldp[j] + rdp[i - 1 - j]);
            }
        }
        return dp;
    }
}

~~~



### LCP64.二叉树灯饰

![image-20240309160214867](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240309160214867.png)

~~~java
~~~



## 二十六.单调栈

<font color=#f00 size=5>**单调栈：保持栈里面的元素是递增或者递减的**</font>

栈中放的是下标，通过一个数组进行映射，

<font color=#f00>**单调栈中递增还是递减**</font>

- 如果是<font color=#f00>**递增**</font>：找的是当前元素后面，<font color=#f00>**第一个比它大的元素**</font>的位置

- 如果是<font color=#f00>**递减**</font>：找的是当前元素后面，<font color=#f00>**第一个比它小的元素**</font>的位置

<font color=#f00 size=5>**十六字真言：及时去掉无用数据，保证栈中元素有序**</font>

- <font color=#f00>**从左到右遍历：可以理解为当前栈中存放的数都是没有找到结果的，如果找到结果的数全都从栈中取出**</font>
- <font color=#f00>**从右到左遍历：可以理解为前一个数，后面可能满足要求的数都存入栈，不满足要求的数都从栈中舍弃**</font>

### 739.每日温度

![image-20240305112150209](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240305112150209.png)

<font color=#00f>**这道题的单调栈的作用，记录一下遍历过的元素，和当前元素进行对比**</font>

单调栈的工作原理：

- 当前遍历的元素T[i]，栈顶的元素T[stack.top()]
- 三种情况，大于小于等于
- 遍历过程：
  - 遍历一个元素，比较当前元素与栈顶元素的大小
    - 如果当前元素比栈顶元素大，那么弹出栈顶元素，比较下一个栈顶元素，直到栈顶元素比当前元素大，记录结果，将当前元素放入栈顶
    - 如果当前元素比栈顶元素小或者相同，那么将当前元素放入栈中



~~~java
public class Solution {
    /**
     * 计算每日温度，给定一个整数数组表示每天的气温，返回一个新的数组，
     * 其中每个元素是需要等待多少天才能遇到比当天更高的气温。
     *
     * @param temperatures 表示每日气温的整数数组
     * @return 返回一个整数数组，表示对于每一天需要等待的天数以找到更高的气温
     */
    public int[] dailyTemperatures(int[] temperatures) {
        // 获取气温数组长度
        int n = temperatures.length;
        
        // 初始化结果数组，初始值为0，将在遍历过程中填充实际需要等待的天数
        int[] daysToWait = new int[n];
        Arrays.fill(daysToWait, 0);

        // 创建单调栈，用于存储气温递减序列的下标
        Stack<Integer> monoStack = new Stack<>();

        // 遍历气温数组
        for (int i = 0; i < n; i++) {
            // 当栈非空且当前气温高于栈顶气温时
            while (!monoStack.isEmpty() && temperatures[i] > temperatures[monoStack.peek()]) {
                // 弹出栈顶气温对应的下标
                int prevDayIndex = monoStack.pop();
                
                // 计算并更新栈顶气温下标对应的等待天数（当前下标与栈顶下标的差值）
                daysToWait[prevDayIndex] = i - prevDayIndex;
            }
            
            // 将当前气温的下标压入单调栈中
            monoStack.push(i);
        }

        // 返回计算得到的等待天数数组
        return daysToWait;
    }
}

~~~

### 496.下一个更大元素I

![image-20240305145823575](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240305145823575.png)

<font color=#00f>**这道题超级绕，不断的转换单调栈中的索引和对应的位置的值**</font>

~~~java
class Solution {
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
         // 初始化结果数组，并填充为-1
        int[] ans = new int[nums1.length];
        Arrays.fill(ans, -1);
        
        // 创建哈希映射，用于快速查找 nums1 中元素的原始下标
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums1.length; i++) {
            map.put(nums1[i], i);
        }
        
        // 创建单调栈，存储遍历 nums2 过程中的元素下标
        Stack<Integer> numStock = new Stack<>();

        // 遍历 nums2
        for (int j = 0; j < nums2.length; j++) {
            // 当栈非空且当前元素大于等于栈顶元素时，弹出栈顶元素并检查是否存在于 nums1 中
            while (!numStock.isEmpty() && nums2[j] > nums2[numStock.peek()]) {
                int temp = numStock.pop();
                if (map.containsKey(nums2[temp])) { // 更正：使用 containsKey 而不是 contain
                    // 更新 nums1 中对应元素的结果值为找到的更大元素在 nums2 中的相对位置
                    ans[map.get(nums2[temp])] = nums2[j];
                }
            }
            
            // 将当前元素的下标压入栈中
            numStock.push(j);
        }

        // 返回计算得到的结果数组
        return ans;
    }
}
~~~

### 503.下一个更大元素II

![image-20240305162229917](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240305162229917.png)

<font color=#00f>**两种方式：遇到这种成环的方式，都可以将数组扩容成二倍**</font>

<font color=#f00>**另一种方法：使用取模的方式，对遍历的i%nums.size()**</font>

~~~java
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        // 初始化结果数组，并填充为-1
        int[] ans = new int[nums.length];
        Arrays.fill(ans, -1);

        // 创建单调栈，存储遍历 nums 过程中的元素下标
        Stack<Integer> numStock = new Stack<>();

        for (int i = 0; i < nums.length * 2; i++) {
            while (!numStock.isEmpty() && nums[i % nums.length] > nums[numStock.peek()]) {
                int temp = numStock.pop();
                ans[temp] = nums[i % nums.length];
            }
            numStock.push(i % nums.length);
        }
        return ans;
    }
}
~~~



### ==42.==接雨水

![image-20240306195745446](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240306195745446.png)

<font color=#00f>**最初使用双指针的方法**</font>

~~~java
class Solution {
    public int trap(int[] height) {
        int ans = 0, left = 0, right = height.length - 1, preMax = 0, sufMax = 0;
        while (left < right) {
            preMax = Math.max(preMax, height[left]);
            sufMax = Math.max(sufMax, height[right]);
            ans += preMax < sufMax ? preMax - height[left++] : sufMax - height[right--];
        }
        return ans;
    }
}

~~~

<font color=#f00>**单调栈的做法，又看不明白了**</font>

~~~java
class Solution {
    public int trap(int[] height) {
        int ans = 0;

        Stack<Integer> heightStock = new Stack<>();
        for(int i = 0; i < height.length; i++){
            while(!heightStock.isEmpty() && height[i] > height[heightStock.peek()]){
                int mid_index = heightStock.pop();
                //注意要在这加一个判断，在找到最后一个元素的时候
                if(heightStock.isEmpty()){
                    break;
                }
                int leftHeight_index = heightStock.peek();
                ans += ((Math.min(height[leftHeight_index],height[i]) - height[mid_index]) * (i- leftHeight_index - 1));

            }
            heightStock.push(i);
        }

        return ans;
    }
}
~~~



![image-20240305111613650](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240305111613650.png)

### 1475.商品折扣后的最终价格

![image-20240306202725773](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240306202725773.png)

<font color=#00f>**就是一道找最近的最小值的题**</font>

~~~java
class Solution {
    public int[] finalPrices(int[] height) {
        int[] ans = new int[height.length];
        for (int i = 0; i < height.length; i++) {
            ans[i] = height[i];
        }

        Stack<Integer> heightStock = new Stack<>();
        for (int i = 0; i < height.length; i++) {
            while (!heightStock.isEmpty() && height[i] <= height[heightStock.peek()]) {
                int zhekou_index = heightStock.pop();
                ans[zhekou_index] = height[zhekou_index] - height[i];
            }
            heightStock.push(i);
        }

        return ans;
    }
}
~~~



### ==901.==股票价格跨度

![image-20240306202810836](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240306202810836.png)

<font color=#f00>**要明白及时去掉栈中无用元素**</font>

~~~java
class StockSpanner {
    private final Deque<int[]> stack = new ArrayDeque<>();
    private int curDay = -1; // 第一个 next 调用算作第 0 天

    public StockSpanner() {
        stack.push(new int[]{-1, Integer.MAX_VALUE}); // 这样无需判断栈为空的情况
    }

    public int next(int price) {
        while (price >= stack.peek()[1]) {
            stack.pop(); // 栈顶数据后面不会再用到了，因为 price 更大
        }
        int ans = ++curDay - stack.peek()[0];
        stack.push(new int[]{curDay, price});
        return ans;
    }
}

~~~



### ==1019.==链表中的下一个更大节点 

![image-20240306204040780](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240306204040780.png)

<font color=#f00>**从右向左遍历：当前栈中存放的是取出左边节点比右边节点大的剩余节点**</font>

<font color=#00f>**递归方法从尾节点遍历**</font>

<font color=#f00>**好好看看这种通过递归反向遍历链表的方法**</font>

~~~java
class Solution {
    private int[] ans;
    private final Deque<Integer> st = new ArrayDeque<>(); // 单调栈（节点值）

    public int[] nextLargerNodes(ListNode head) {
        dfs(head,0);
        return ans;
    }

//当前节点的下一个最大节点值的位置和
    private void dfs(ListNode node, int i){
        if(node == null){
            ans = new int[i];//当递归到尾节点时，找到了链表的长度，创建一个长度为链表长度的数组
            return;
        }
        dfs(node.next, i+1);//递归到链表的结尾
        while(!st.isEmpty() && st.peek() <= node.val){
            st.pop();//清除数组中的无用数据
        }
        if(!st.isEmpty()){
            ans[i] = st.peek();//栈顶就是i个节点的下一个更大元素
        }
        st.push(node.val);

    }

~~~

<font color=#00f>**不想递归就使用反转链表的方式从头结点进行遍历**</font>

~~~java
class Solution {
    private int n;

    public int[] nextLargerNodes(ListNode head) {
        head = reverseList(head);
        var ans = new int[n];
        var st = new ArrayDeque<Integer>(); // 单调栈（节点值）
        for (var cur = head; cur != null; cur = cur.next) {
            while (!st.isEmpty() && st.peek() <= cur.val)
                st.pop(); // 弹出无用数据
            ans[--n] = st.isEmpty() ? 0 : st.peek();
            st.push(cur.val);
        }
        return ans;
    }

    // 206. 反转链表
    public ListNode reverseList(ListNode head) {
        ListNode pre = null, cur = head;
        while (cur != null) {
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
            ++n;
        }
        return pre;
    }
}
~~~





<font color=#f00>**从左向右遍历：当前栈中存放的是没有找到更大节点值的节点**</font>

~~~java
class Solution {
    public int[] nextLargerNodes(ListNode head) {
        int n = 0;
        for (var cur = head; cur != null; cur = cur.next)
            ++n; // 确定返回值的长度
        var ans = new int[n];
        var waitAnsNode = new ArrayDeque<int[]>(); // 单调栈（节点值，节点下标）
        int i=0;
        for(var cur = head; cur != null; cur = cur.next){
            while(!waitAnsNode.isEmpty() && waitAnsNode.peek()[0] < cur.val){
                ans[waitAnsNode.pop()[1]] = cur.val;//用当前节点值更新答案
            }
            waitAnsNode.push(new int[]{cur.val, i++});
        }
        return ans;
    }
}

~~~



### ==1944.==队列中可以看到的人数

![image-20240306204114613](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240306204114613.png)

<font color=#00f>**思路，倒序遍历，当左边的人比右边的人高的时候，那么更左边的人将看不到右边的矮的人，所以此时将右边矮的人出栈**</font>

~~~java
// 更快的写法见数组版本
class Solution {
    public int[] canSeePersonsCount(int[] heights) {
        int n = heights.length;
        int[] ans = new int[n];
        Deque<Integer> st = new ArrayDeque<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!st.isEmpty() && st.peek() < heights[i]) {//这个地方写不写等号都是对的
                st.pop();
                ans[i]++;
            }
            if (!st.isEmpty()) { // 还可以再看到一个人
                ans[i]++;
            }
            st.push(heights[i]);
        }
        return ans;
    }
}

~~~

### 84.柱状图中最大的矩形

![image-20240307092117218](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240307092117218.png)

<font color=#00f>**算法：维护两个数组，一个数组对应位置存放当前元素第一个左边比它小的元素，另一个数组存放当前元素右边第一个比它小的元素位置**</font>

~~~JAVA
class Solution {
    public int largestRectangleArea(int[] hs) {
        int n = hs.length;
        int[] l = new int[n], r = new int[n];
        Arrays.fill(l, -1); Arrays.fill(r, n);
        Deque<Integer> d = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (!d.isEmpty() && hs[d.peekLast()] > hs[i]) r[d.pollLast()] = i;
            d.addLast(i);
        }
        d.clear();
        for (int i = n - 1; i >= 0; i--) {
            while (!d.isEmpty() && hs[d.peekLast()] > hs[i]) l[d.pollLast()] = i;
            d.addLast(i);
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            int t = hs[i], a = l[i], b = r[i];
            ans = Math.max(ans, (b - a - 1) * t);
        }
        return ans;
    }
}

~~~

<font color=#00f>**自己写的两种不同的单调栈思路**</font>

~~~java
class Solution {
    public int largestRectangleArea(int[] heights) {
        // 求维护当前元素右边第一个比它小的数组,使用从左到右单调栈思想
        int[] rightMinHeights = new int[heights.length];
        Deque<Integer> rightStack = new ArrayDeque<>();
        for (int i = heights.length - 1; i >= 0; i--) {
            while (!rightStack.isEmpty() && heights[i] < heights[rightStack.peek()]) {
                int temp = rightStack.pop();
            }
            if (!rightStack.isEmpty()) {
                rightMinHeights[i] = rightStack.peek();
            } else {
                rightMinHeights[i] = heights.length - 1;
            }
            rightStack.push(i);
        }
        // 求维护当前元素左边第一个比它小的数组，是由从右到左的单调栈思想，和上面使用不同思路
        int[] leftMinHeights = new int[heights.length];
        Deque<Integer> leftStack = new ArrayDeque<>();
        for (int i = heights.length - 1; i >= 0; i--) {
            while (!leftStack.isEmpty() && heights[leftStack.peek()] > heights[i]) {
                leftMinHeights[leftStack.pop()] = i;
            }
            leftStack.push(i);
        }
        int ans = 0;
        for (int i = 0; i < heights.length; i++) {
            int t = heights[i], a = leftMinHeights[i], b = rightMinHeights[i];
            ans = Math.max(ans, (b - a - 1) * t);
        }
        return ans;
    }
}
~~~



### 85.最大矩形

![image-20240307101828155](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240307101828155.png)

<font color=#00f>**思路：将这道题转化为上一道**</font>

![image-20240307102705544](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240307102705544.png)

~~~java
class Solution {
    public int maximalRectangle(char[][] mat) {
        int n = mat.length, m = mat[0].length, ans = 0;
        int[][] sum = new int[n + 10][m + 10];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                sum[i][j] = mat[i - 1][j - 1] == '0' ? 0 : sum[i - 1][j] + 1;
            }
        }
        int[] l = new int[m + 10], r = new int[m + 10];
        for (int i = 1; i <= n; i++) {
            int[] cur = sum[i];
            Arrays.fill(l, 0); Arrays.fill(r, m + 1);
            Deque<Integer> d = new ArrayDeque<>();
            for (int j = 1; j <= m; j++) {
                while (!d.isEmpty() && cur[d.peekLast()] > cur[j]) r[d.pollLast()] = j;
                d.addLast(j);
            }
            d.clear();
            for (int j = m; j >= 1; j--) {
                while (!d.isEmpty() && cur[d.peekLast()] > cur[j]) l[d.pollLast()] = j;
                d.addLast(j);
            }
            for (int j = 1; j <= m; j++) ans = Math.max(ans, cur[j] * (r[j] - l[j] - 1));
        }
        return ans;
    }
}


~~~



## 二十七.单调队列

### 239.滑动窗口最大值

![image-20240312141102166](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240312141102166.png)

~~~java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int[] ans = new int[n - k + 1];
        Deque<Integer> q = new ArrayDeque<>(); // 双端队列
        for (int i = 0; i < n; i++) {
            // 1. 入
            while (!q.isEmpty() && nums[q.getLast()] <= nums[i]) {
                q.removeLast(); // 维护 q 的单调性
            }
            q.addLast(i); // 入队
            // 2. 出
            if (i - q.getFirst() >= k) { // 队首已经离开窗口了
                q.removeFirst();
            }
            // 3. 记录答案
            if (i >= k - 1) {
                // 由于队首到队尾单调递减，所以窗口最大值就是队首
                ans[i - k + 1] = nums[q.getFirst()];
            }
        }
        return ans;
    }
}

~~~





![image-20240305111347525](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240305111347525.png)

### ==1438.==绝对值不超过限制的最长连续子数组

![image-20240312142801448](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240312142801448.png)

<font color=#00f>**基本思想是双指针，通过双端队列维护窗口内的最大值和最小值，如果当前窗口中最大值最小值之差大于限制，就不断将左指针右移，如果移动过程中发现最大值或最小值就是当前数列中的最左侧的值，就将最大值或最小值队列中的值给移除，直到最大值最小值之差小于限值**</font>

~~~java
public int longestSubarray(int[] nums, int limit) {
    Deque<Integer> maxQueue = new ArrayDeque<>();
    Deque<Integer> minQueue = new ArrayDeque<>();
    int l = 0, r = 0, res = 0;
    while (r < nums.length) {
        while (!maxQueue.isEmpty() && nums[r] > maxQueue.peekLast()) 
            maxQueue.removeLast();
        while (!minQueue.isEmpty() && nums[r] < minQueue.peekLast()) 
            minQueue.removeLast();
        maxQueue.add(nums[r]);
        minQueue.add(nums[r]);
        r++;
        while (maxQueue.peek() - minQueue.peek() > limit) {
            if (maxQueue.peek() == nums[l]) maxQueue.remove();
            if (minQueue.peek() == nums[l]) minQueue.remove();   
            l += 1;
        }
        res = Math.max(res, r - l);
    }
    return res;
}

~~~



### 2398.预算内的最多机器人数目



### ==862.==和至少为K的最短子数组

![image-20240313112612768](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240313112612768.png)



![862-1-2.png](./Java重写：灵茶山艾府——基础算法精讲.assets/1666668814-ikkWBN-862-1-2.png)![862-2-3.png](./Java重写：灵茶山艾府——基础算法精讲.assets/1666669250-KypIVI-862-2-3.png)

~~~java
class Solution {
    public int shortestSubarray(int[] nums, int k) {
        int n = nums.length, ans = n + 1;
        var s = new long[n + 1];
        for (var i = 0; i < n; ++i)
            s[i + 1] = s[i] + nums[i]; // 计算前缀和
        var q = new ArrayDeque<Integer>();
        for (var i = 0; i <= n; ++i) {
            var curS = s[i];
            while (!q.isEmpty() && curS - s[q.peekFirst()] >= k)
                ans = Math.min(ans, i - q.pollFirst()); // 优化一：满足条件之后记录答案，并将此时的最小值弹出
            while (!q.isEmpty() && s[q.peekLast()] >= curS)
                q.pollLast(); // 优化二
            q.addLast(i);
        }
        return ans > n ? -1 : ans;
    }
}
~~~





### ==1499.==满足不等式的最大值

![image-20240313112633200](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240313112633200.png)

~~~java
class Solution {
    public int findMaxValueOfEquation(int[][] points, int k) {
        int ans = Integer.MIN_VALUE;
        var q = new ArrayDeque<int[]>();
        for (var p : points) {
            int x = p[0], y = p[1];
            while (!q.isEmpty() && q.peekFirst()[0] < x - k) // 队首超出范围
                q.pollFirst(); // 弹它！
            if (!q.isEmpty())
                ans = Math.max(ans, x + y + q.peekFirst()[1]); // 加上最大的 yi-xi
            while (!q.isEmpty() && q.peekLast()[1] <= y - x) // 队尾不如新来的强
                q.pollLast(); // 弹它！
            q.addLast(new int[]{x, y - x});
        }
        return ans;
    }
}

~~~



### 1696.跳跃游戏VI

![image-20240313093001421](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240313093001421.png)

<font color=#00f>**递归代码**</font>

~~~java
// 会超时的递归代码
class Solution {
    public int maxResult(int[] nums, int k) {
        return dfs(nums.length - 1, nums, k);
    }

    private int dfs(int i, int[] nums, int k) {
        if (i == 0) {
            return nums[0];
        }
        int mx = Integer.MIN_VALUE;
        for (int j = Math.max(i - k, 0); j < i; j++) {
            mx = Math.max(mx, dfs(j, nums, k));
        }
        return mx + nums[i];
    }
}

~~~

<font color=#00f>**递推代码**</font>

~~~java
class Solution {
    public int maxResult(int[] nums, int k) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[0] = nums[0];
        for (int i = 1; i < n; i++) {
            int mx = Integer.MAX_VALUE;
            for (int j = Math.max(i - k, 0); j < i; j++) {
                mx = Math.max(mx, dp[j]);
            }
            dp[i] = mx + nums[i];
        }
        return dp[n - 1];
    }
}

~~~

<font color=#00f>**单调队列优化**</font>

~~~java
class Solution {
    public int maxResult(int[] nums, int k) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[0] = nums[0];
        Deque<Integer> q = new ArrayDeque<>();
        q.add(0);
        for (int i = 1; i < n; i++) {
            // 1. 出
            if (q.peekFirst() < i - k) {
                q.pollFirst();
            }
            // 2. 转移
            dp[i] = dp[q.peekFirst()] + nums[i];
            // 3. 入
            while (!q.isEmpty() && dp[i] >= dp[q.peekLast()]) {
                q.pollLast();
            }
            q.add(i);
        }
        return dp[n - 1];
    }
}

~~~

<font color=#f00>**自己写的代码，但是存在问题，在队首元素弹出的问题上**</font>

~~~java
class Solution {
    public int maxResult(int[] nums, int k) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[0] = nums[0];
        Deque<Integer> q = new ArrayDeque<>();
        q.add(0);
        for (int i = 1; i < n; i++) {
            // 3. 入
            while (!q.isEmpty() && nums[q.peekLast()] < nums[i]) {
                q.pollLast();
            }
            q.add(i);
            // 2. 出,超出队列范围
            if (q.peekFirst() < i - k) {
                q.pollFirst();
            }
            dp[i] = dp[q.peekFirst()] + nums[i];
            System.out.println(Arrays.toString(dp));
        }
        return dp[n - 1];
    }
}
~~~

![image-20240313102217173](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240313102217173.png)

![image-20240313103138575](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240313103138575.png)

<font color=#f00>**空间优化**</font>

~~~java
class Solution {
    public int maxResult(int[] nums, int k) {
        int n = nums.length;
        Deque<Integer> q = new ArrayDeque<>();
        q.add(0);
        for (int i = 1; i < n; i++) {
            // 1. 出
            if (q.peekFirst() < i - k) {
                q.pollFirst();
            }
            // 2. 转移
            nums[i] += nums[q.peekFirst()];
            // 3. 入
            while (!q.isEmpty() && nums[i] >= nums[q.peekLast()]) {
                q.pollLast();
            }
            q.add(i);
        }
        return nums[n - 1];
    }
}

~~~



### 2944.购买水果需要的最少金币数

![image-20240313094909918](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240313094909918.png)

<font color=#00f>**记忆化搜索**</font>

~~~java
class Solution {
    public int minimumCoins(int[] prices) {
        int n = prices.length;
        int[] memo = new int[(n + 1) / 2];
        return dfs(1, prices, memo);
    }
	// 获取第i个及其后面的水果所需的最少花费
    private int dfs(int i, int[] prices, int[] memo) {
        if (i * 2 >= prices.length) {
            return prices[i - 1]; // i 从 1 开始
        }
        if (memo[i] != 0) { // 之前算过
            return memo[i];
        }
        int res = Integer.MAX_VALUE;
        for (int j = i + 1; j <= i * 2 + 1; j++) {
            res = Math.min(res, dfs(j, prices, memo));
        }
        return memo[i] = res + prices[i - 1]; // 记忆化
    }
}

~~~

<font color=#00f>**递推**</font>

~~~java
class Solution {
    public int minimumCoins(int[] prices) {
        int n = prices.length;
        for (int i = (n + 1) / 2 - 1; i > 0; i--) {
            int mn = Integer.MAX_VALUE;
            for (int j = i; j <= i * 2; j++) {
                mn = Math.min(mn, prices[j]);
            }
            prices[i - 1] += mn;
        }
        return prices[0];
    }
}

~~~

<font color=#00f>**单调队列优化**</font>

~~~java
class Solution {
    public int minimumCoins(int[] prices) {
        int n = prices.length;
        Deque<int[]> q = new ArrayDeque<>();
        q.addLast(new int[]{n + 1, 0}); // 哨兵
        for (int i = n; i > 0; i--) {
            while (q.peekLast()[0] > i * 2 + 1) { // 右边离开窗口
                q.pollLast();
            }
            int f = prices[i - 1] + q.peekLast()[1];
            while (f <= q.peekFirst()[1]) {
                q.pollFirst();
            }
            q.addFirst(new int[]{i, f});  // 左边进入窗口
        }
        return q.peekFirst()[1];
    }
}

~~~

## 二十八.矩阵

### 36.有效的数独

![image-20240317204900536](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240317204900536.png)

~~~java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        //这三个矩阵一个是判断row[i][shuzhi]这个位置是否有数值
        boolean[][] row = new boolean[10][10], col = new boolean[10][10], area = new boolean[10][10];        
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                int c = board[i][j];
                if (c == '.') continue;
                int u = c - '0';
                //这个地方是重点！！！
                int idx = i / 3 * 3 + j / 3;
                if (row[i][u] || col[j][u] || area[idx][u]) return false;
                row[i][u] = col[j][u] = area[idx][u] = true;
            }
        }
        return true;
    }
}

~~~

<font color=#f00>**位运算**</font>

~~~java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        int[] row = new int[10], col = new int[10], area = new int[10];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char c = board[i][j];
                if (c == '.') continue;
                int u = c - '0';
                int idx = i / 3 * 3 + j / 3;
                if ((((row[i] >> u) & 1) == 1) || (((col[j] >> u) & 1) == 1) || (((area[idx] >> u) & 1) == 1)) return false;
                row[i] |= (1 << u);
                col[j] |= (1 << u);
                area[idx] |= (1 << u);
            }
        }
        return true;
    }
}

~~~

## 三十九.图

### 743.网络延迟时间

![image-20240421162232019](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240421162232019.png)

<font color=#00f size=6>**用到Dijkstra算法**</font>

![image-20240421162016732](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240421162016732.png)

![image-20240421162034623](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240421162034623.png)

~~~java
class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        final int INF = Integer.MAX_VALUE / 2; // 防止加法溢出
        int[][] g = new int[n][n]; // 邻接矩阵
        for (int[] row : g) {
            Arrays.fill(row, INF);
        }
        for (int[] t : times) {
            g[t[0] - 1][t[1] - 1] = t[2];
        }

        int maxDis = 0;
        int[] dis = new int[n];
        Arrays.fill(dis, INF);
        dis[k - 1] = 0;
        boolean[] done = new boolean[n];
        while (true) {
            int x = -1;
            for (int i = 0; i < n; i++) {
                if (!done[i] && (x < 0 || dis[i] < dis[x])) {
                    x = i;
                }
            }
            if (x < 0) {
                return maxDis; // 最后一次算出的最短路就是最大的
            }
            if (dis[x] == INF) { // 有节点无法到达
                return -1;
            }
            maxDis = dis[x]; // 求出的最短路会越来越大
            done[x] = true; // 最短路长度已确定（无法变得更小）
            for (int y = 0; y < n; y++) {
                // 更新 x 的邻居的最短路
                dis[y] = Math.min(dis[y], dis[x] + g[x][y]);
            }
        }
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/network-delay-time/solutions/2668220/liang-chong-dijkstra-xie-fa-fu-ti-dan-py-ooe8/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

![image-20240421162111161](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240421162111161.png)

~~~java
class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        List<int[]>[] g = new ArrayList[n]; // 邻接表
        Arrays.setAll(g, i -> new ArrayList<>());
        for (int[] t : times) {
            g[t[0] - 1].add(new int[]{t[1] - 1, t[2]});
        }

        int maxDis = 0;
        int left = n; // 未确定最短路的节点个数
        int[] dis = new int[n];
        Arrays.fill(dis, Integer.MAX_VALUE);
        dis[k - 1] = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> (a[0] - b[0]));
        pq.offer(new int[]{0, k - 1});
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int dx = p[0];
            int x = p[1];
            if (dx > dis[x]) { // x 之前出堆过
                continue;
            }
            maxDis = dx; // 求出的最短路会越来越大
            left--;
            for (int[] e : g[x]) {
                int y = e[0];
                int newDis = dx + e[1];
                if (newDis < dis[y]) {
                    dis[y] = newDis; // 更新 x 的邻居的最短路
                    pq.offer(new int[]{newDis, y});
                }
            }
        }
        return left == 0 ? maxDis : -1;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/network-delay-time/solutions/2668220/liang-chong-dijkstra-xie-fa-fu-ti-dan-py-ooe8/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



## 四十.岛屿类问题

<font color=#f00 size=5>**网格类DFS遍历框架**</font>

![image-20240418102912635](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418102912635.png)

![image-20240418102924292](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418102924292.png)

<font color=#f00 size=5>**基本框架**</font>

~~~java
void dfs(int[][] grid, int r, int c) {
    // 判断 base case
    // 如果坐标 (r, c) 超出了网格范围，直接返回
    if (!inArea(grid, r, c)) {
        return;
    }
    // 访问上、下、左、右四个相邻结点
    dfs(grid, r - 1, c);
    dfs(grid, r + 1, c);
    dfs(grid, r, c - 1);
    dfs(grid, r, c + 1);
}

// 判断坐标 (r, c) 是否在网格中
boolean inArea(int[][] grid, int r, int c) {
    return 0 <= r && r < grid.length 
        	&& 0 <= c && c < grid[0].length;
}

作者：nettee
链接：https://leetcode.cn/problems/number-of-islands/solutions/211211/dao-yu-lei-wen-ti-de-tong-yong-jie-fa-dfs-bian-li-/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

<font color=#f00 size=5>**网格遍历和树遍历的方法不同点**</font>

网格遍历有可能一直在兜圈子

![image-20240418103347649](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418103347649.png)

~~~java
void dfs(int[][] grid, int r, int c) {
    // 判断 base case
    if (!inArea(grid, r, c)) {
        return;
    }
    // 如果这个格子不是岛屿，直接返回
    if (grid[r][c] != 1) {
        return;
    }
    grid[r][c] = 2; // 将格子标记为「已遍历过」
    
    // 访问上、下、左、右四个相邻结点
    dfs(grid, r - 1, c);
    dfs(grid, r + 1, c);
    dfs(grid, r, c - 1);
    dfs(grid, r, c + 1);
}

// 判断坐标 (r, c) 是否在网格中
boolean inArea(int[][] grid, int r, int c) {
    return 0 <= r && r < grid.length 
        	&& 0 <= c && c < grid[0].length;
}

作者：nettee
链接：https://leetcode.cn/problems/number-of-islands/solutions/211211/dao-yu-lei-wen-ti-de-tong-yong-jie-fa-dfs-bian-li-/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



### 695.岛屿的最大面积

![image-20240418104557146](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418104557146.png)

~~~java
public int maxAreaOfIsland(int[][] grid) {
    int res = 0;
    for (int r = 0; r < grid.length; r++) {
        for (int c = 0; c < grid[0].length; c++) {
            if (grid[r][c] == 1) {
                int a = area(grid, r, c);
                res = Math.max(res, a);
            }
        }
    }
    return res;
}

int area(int[][] grid, int r, int c) {
    if (!inArea(grid, r, c)) {
        return 0;
    }
    if (grid[r][c] != 1) {
        return 0;
    }
    grid[r][c] = 2;
    
    return 1 
        + area(grid, r - 1, c)
        + area(grid, r + 1, c)
        + area(grid, r, c - 1)
        + area(grid, r, c + 1);
}

boolean inArea(int[][] grid, int r, int c) {
    return 0 <= r && r < grid.length 
        	&& 0 <= c && c < grid[0].length;
}

~~~



### 200.岛屿数量

![image-20240418105000774](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418105000774.png)

~~~java
class Solution {
    public int numIslands(char[][] grid) {
        int res = 0;
        for (int r = 0; r < grid.length; r++) {
            for (int c = 0; c < grid[0].length; c++) {
                if (grid[r][c] == '1') {
                    dfs(grid, r, c);
                    res += 1;
                }
            }
        }
        return res;
    }

    void dfs(char[][] grid, int r, int c) {
        if (!inArea(grid, r, c)) {
            return;
        }
        if (grid[r][c] != '1') {
            return;
        }
        grid[r][c] = '2';

        // 访问上、下、左、右四个相邻结点
        dfs(grid, r - 1, c);
        dfs(grid, r + 1, c);
        dfs(grid, r, c - 1);
        dfs(grid, r, c + 1);
    }

    boolean inArea(char[][] grid, int r, int c) {
        return 0 <= r && r < grid.length && 0 <= c && c < grid[0].length;
    }
}
~~~

### 463.岛屿的周长

![image-20240418110453074](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418110453074.png)

~~~java
class Solution {
    public int islandPerimeter(int[][] grid) {
        for (int r = 0; r < grid.length; r++) {
            for (int c = 0; c < grid[0].length; c++) {
                if (grid[r][c] == 1) {
                    // 题目限制只有一个岛屿，计算一个即可
                    return dfs(grid, r, c);
                }
            }
        }
        return 0;
    }

    int dfs(int[][] grid, int r, int c) {
        // 函数因为「坐标 (r, c) 超出网格范围」返回，对应一条黄色的边
        if (!inArea(grid, r, c)) {
            return 1;
        }
        // 函数因为「当前格子是海洋格子」返回，对应一条蓝色的边
        if (grid[r][c] == 0) {
            return 1;
        }
        // 函数因为「当前格子是已遍历的陆地格子」返回，和周长没关系
        if (grid[r][c] != 1) {
            return 0;
        }
        grid[r][c] = 2;
        return dfs(grid, r - 1, c)
                + dfs(grid, r + 1, c)
                + dfs(grid, r, c - 1)
                + dfs(grid, r, c + 1);
    }

    // 判断坐标 (r, c) 是否在网格中
    boolean inArea(int[][] grid, int r, int c) {
        return 0 <= r && r < grid.length
                && 0 <= c && c < grid[0].length;
    }

}
~~~





## 四十二.前缀和

<font color=#f00 size=6>**基础部分**</font>

### 2559.统计范围内的元音字符串数

![image-20240502203007915](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240502203007915.png)

~~~java
class Solution {
    Set<Character> set = new HashSet<>(Arrays.asList('a','e','i','o','u'));

    public int[] vowelStrings(String[] words, int[][] queries) {
        int n = words.length;
        int[] sum = new int[n + 1];
        sum[1] = panduan(words[0])? 1 : 0;
        for(int i=2; i<= n; i++){
            if(panduan(words[i-1])){
                sum[i] = sum[i-1] + 1;
            }else{
                sum[i] = sum[i-1];
            }
        }
        int nn = queries.length;
        int[] res = new int[nn];
        for(int i=0; i<nn; i++){
            res[i] = sum[queries[i][1] + 1] - sum[queries[i][0]];
        }
        return res;
    }

    private boolean panduan(String str){
        int n = str.length();
        if(set.contains(str.charAt(0)) && set.contains(str.charAt(n-1))){
            return true;
        }else{
            return false;
        }
    }
}
~~~

### 1744.你能在你最喜欢的那天吃到你最喜欢的糖果吗

![image-20240503084422858](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240503084422858.png)

~~~java
class Solution {
    public boolean[] canEat(int[] cs, int[][] qs) {
        int n = qs.length, m = cs.length;
        boolean[] ans = new boolean[n];
        long[] sum = new long[m + 1];
        for (int i = 1; i <= m; i++) sum[i] = sum[i - 1] + cs[i - 1];
        for (int i = 0; i < n; i++) {
            int t = qs[i][0], d = qs[i][1] + 1, c = qs[i][2];
            long a = sum[t] / c + 1, b = sum[t + 1];
            ans[i] = a <= d && d <= b;
        }
        return ans;
    }
}

~~~

<font color=#f00 size=6>**前缀和与哈希表**</font>

### 930.和相同的二元子数组

![image-20240503085127353](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240503085127353.png)

~~~java
// 获取数组长度
int n = nums.length;

// 初始化一个前缀和数组，sum[i]表示nums数组前i个元素的和
int[] sum = new int[n + 1];
// 计算前缀和数组
for (int i = 1; i <= n; i++) {
    sum[i] = sum[i - 1] + nums[i - 1];
}

// 使用哈希表记录每个前缀和出现的次数，初始化为sum[0]=0出现1次
Map<Integer, Integer> map = new HashMap<>();
map.put(0, 1);

// 初始化计数器，用于记录符合条件的子数组数量
int ans = 0;

// 遍历原数组，利用前缀和计算子数组和
for (int i = 0; i < n; i++) {
    // 当前位置i的前缀和
    int r = sum[i + 1];
    // 目标值t相对于当前前缀和的差值，即寻找是否有前缀和等于r-t
    int l = r - t;
    
    // 查找是否存在前缀和等于l，如果有，则加上其出现次数到答案中
    ans += map.getOrDefault(l, 0);
    
    // 将当前前缀和r的计数加1，存入哈希表
    map.put(r, map.getOrDefault(r, 0) + 1);
}

// 返回符合条件的子数组数量
return ans;


~~~

<font color=#00f>**这道题如果让自己写，肯定是需要两边循环，来进行遍历，时间复杂度大大提高，代码如下，不能使用**</font>

~~~java
class Solution {
    public int numSubarraysWithSum(int[] nums, int goal) {
        int n  = nums.length;
        int[] sum = new int[n + 1];
        sum[1] = nums[0];
        for(int i = 2; i <= n ;i++){
            sum[i] = sum[i - 1] + nums[i - 1];
        }
        int res = 0;
        for(int i = 1; i<=n;i++){
            for(int j = 0; j < i; j++){
                if(sum[i] - sum[j] == goal){
                    res++;
                }
            }
        }
        return res;
    }
}
~~~

### 1524.和为奇数的子数组数目

![image-20240503093357199](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240503093357199.png)

~~~java
class Solution {
    public static final int MOD = 1000000007;
    public int numOfSubarrays(int[] arr) {

        int n = arr.length;
        int[] sum = new int[n+1];
        sum[1] = arr[0];
        //两个变量，分别记录当前前缀和中的奇偶数个数
        int even = 1;//偶数个数，因为空前缀和为0，所以初始化为1
        int odd = 0;//奇数个数
        int ans = 0;
        for(int i = 1 ; i <= n ; i++){
            sum[i] = (sum[i-1] + arr[i-1])%MOD;
            ans = (ans + (sum[i]%2 == 0 ? odd : even))%MOD;
            if(sum[i]%2 == 0){
                even++;
            }else{
                odd++;
            }
        }
        return ans;
    }
}
~~~

### ==1590.==使数组和能被P整除

![image-20240504152511004](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504152511004.png)

![image-20240503095113757](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240503095113757.png)

~~~java
class Solution {
    public int minSubarray(int[] nums, int p) {
        int n = nums.length, ans = n;
        var s = new int[n + 1];
        for (int i = 0; i < n; ++i)
            s[i + 1] = (s[i] + nums[i]) % p;
        int x = s[n];
        if (x == 0) return 0; // 移除空子数组（这行可以不要）

        var last = new HashMap<Integer, Integer>();
        for (int i = 0; i <= n; ++i) {
            last.put(s[i], i);
            // 如果不存在，-n 可以保证 i-j >= n
            int j = last.getOrDefault((s[i] - x + p) % p, -n);
            ans = Math.min(ans, i - j);
        }
        return ans < n ? ans : -1;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/make-sum-divisible-by-p/solutions/2158435/tao-lu-qian-zhui-he-ha-xi-biao-pythonjav-rzl0/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



### 面试题17.05.字母与数组

![image-20240504144139139](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504144139139.png)

![image-20240504151245628](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504151245628.png)

~~~java
class Solution {
    public String[] findLongestSubarray(String[] array) {
        int n = array.length;
        var s = new int[n + 1]; // 前缀和
        for (int i = 0; i < n; ++i)
            s[i + 1] = s[i] + (array[i].charAt(0) >> 6 & 1) * 2 - 1;

        int begin = 0, end = 0; // 符合要求的子数组 [begin,end)
        var first = new HashMap<Integer, Integer>();
        for (int i = 0; i <= n; ++i) {
            int j = first.getOrDefault(s[i], -1);
            if (j < 0) // 首次遇到 s[i]
                first.put(s[i], i);
            else if (i - j > end - begin) { // 更长的子数组
                begin = j;
                end = i;
            }
        }

        var sub = new String[end - begin];
        System.arraycopy(array, begin, sub, 0, sub.length);
        return sub;
    }
}
~~~



<font color=#f00 size=6>**距离和**</font>

### ==2602.==使数组元素全部相等的最少操作

![image-20240504153117812](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504153117812.png)

<font color=#00f>**排序，前缀和，二分查找**</font>

![image-20240504153213419](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504153213419.png)

~~~java
class Solution {
    public List<Long> minOperations(int[] nums, int[] queries) {
        Arrays.sort(nums);
        int n = nums.length;
        var sum = new long[n + 1]; // 前缀和
        for (int i = 0; i < n; ++i)
            sum[i + 1] = sum[i] + nums[i];

        var ans = new ArrayList<Long>(queries.length);
        for (int q : queries) {
            int j = lowerBound(nums, q);
            long left = (long) q * j - sum[j]; // 蓝色面积
            long right = sum[n] - sum[j] - (long) q * (n - j); // 绿色面积
            ans.add(left + right);
        }
        return ans;
    }

    // 见 https://www.bilibili.com/video/BV1AP41137w7/
    private int lowerBound(int[] nums, int target) {
        int left = -1, right = nums.length; // 开区间 (left, right)
        while (left + 1 < right) { // 区间不为空
            // 循环不变量：
            // nums[left] < target
            // nums[right] >= target
            int mid = left + (right - left) / 2;
            if (nums[mid] < target)
                left = mid; // 范围缩小到 (mid, right)
            else
                right = mid; // 范围缩小到 (left, mid)
        }
        return right;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/minimum-operations-to-make-all-array-elements-equal/solutions/2191417/yi-tu-miao-dong-pai-xu-qian-zhui-he-er-f-nf55/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

~~~java
class Solution {
    public List<Long> minOperations(int[] nums, int[] queries) {
        Arrays.sort(nums);
        int n = nums.length;
        var sum = new long[n + 1]; // 前缀和
        for (int i = 0; i < n; ++i){
            sum[i + 1] = sum[i] + nums[i];
        }

        List<Long> ans = new ArrayList<>(queries.length);
        for(int q:queries){
            int j = lowerBound(nums,q);
            long left = (long) q * j - sum[j];
            long right = sum[n] - sum[j] - (long) q * (n-j);
            ans.add(left + right);
        }
        return ans;
            
    }

    public int lowerBound(int[] nums, int target){
        //使用闭区间进行查找
        int left = 0;
        int right = nums.length - 1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] < target){
                left = mid + 1;
            }else{
                right = mid - 1;
            }
        }
        return left;
    }
}
~~~

### 1685.有序数组中差绝对值之和

![image-20240504154101739](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504154101739.png)

~~~java
class Solution {
    public int[] getSumAbsoluteDifferences(int[] nums) {
        int n = nums.length;
        int[] sum = new int[n + 1];
        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + nums[i];
        }
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            int cur = nums[i];
            result[i] = cur * (i + 1) - sum[i + 1] + (sum[n] - sum[i + 1]) - cur * (n - i - 1);
        }
        return result;
    }
}
~~~

### 2968.执行操作使频率分数最大

![image-20240504161435717](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504161435717.png)

~~~java
class Solution {
    public int maxFrequencyScore(int[] nums, long k) {
        Arrays.sort(nums);

        int n = nums.length;
        long[] s = new long[n + 1];
        for (int i = 0; i < n; i++) {
            s[i + 1] = s[i] + nums[i];
        }

        int ans = 0, left = 0;
        for (int i = 0; i < n; i++) {
            while (distanceSum(s, nums, left, (left + i) / 2, i) > k) {
                left++;
            }
            ans = Math.max(ans, i - left + 1);
        }
        return ans;
    }

    // 把 nums[l] 到 nums[r] 都变成 nums[i]
    long distanceSum(long[] s, int[] nums, int l, int i, int r) {
        long left = (long) nums[i] * (i - l) - (s[i] - s[l]);
        long right = s[r + 1] - s[i + 1] - (long) nums[i] * (r - i);
        return left + right;
    }
}
~~~

<font color=#f00 size=6>**前缀异或和**</font>

### 1310.子数组异或查询

![image-20240504161525142](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504161525142.png)

<font color=#f00>**注意一个点，异或的逆运算还是异或**</font>

~~~java
class Solution {
    public int[] xorQueries(int[] arr, int[][] queries) {
        int n = arr.length;
        int m = queries.length;
        int[] sum = new int[n+1];
        for(int i = 0; i <n;i++){
            sum[i+1] = sum[i] ^ arr[i];
        }
        int[] ans = new int[m];
        for(int i = 0 ;i<m;i++){
            int l = queries[i][0] + 1;
            int r = queries[i][1] + 1;
            ans[i] = sum[r] ^ sum[l-1];
        }
        return ans;
    }
}
~~~

### 1177.构件回文串监测

![image-20240504164745037](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504164745037.png)

~~~java
class Solution {
    public List<Boolean> canMakePaliQueries(String s, int[][] queries) {
        int n = s.length();
        var sum = new int[n + 1][26];
        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i].clone();
            sum[i + 1][s.charAt(i) - 'a']++;
        }

        var ans = new ArrayList<Boolean>(queries.length); // 预分配空间
        for (var q : queries) {
            int left = q[0], right = q[1], k = q[2], m = 0;
            for (int j = 0; j < 26; j++)
                m += (sum[right + 1][j] - sum[left][j]) % 2; // 奇数+1，偶数+0
            ans.add(m / 2 <= k);
        }
        return ans;
    }
}
~~~

~~~java
class Solution {
    public List<Boolean> canMakePaliQueries(String s, int[][] queries) {
        int n = s.length();
        int[][] sum = new int[n + 1][26];
        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i].clone();
            sum[i + 1][s.charAt(i) - 'a']++;
        }
        List<Boolean> ans = new ArrayList<>(queries.length);
        for (int[] q : queries) {
            int left = q[0], right = q[1], k = q[2];
            int temp = 0;
            for (int j = 0; j < 26; j++) {
                temp += (sum[right + 1][j] - sum[left][j]) % 2;
            }
            ans.add(temp / 2 <= k);
        }
        return ans;
    }
}
~~~

<font color=#f00 size=6>**其他一维前缀和**</font>

### 1895.最大的幻方

![image-20240504170603586](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504170603586.png)

~~~java
class Solution {
    private int[][] rowsum;
    private int[][] colsum;

    public int largestMagicSquare(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        rowsum = new int[m + 1][n + 1];
        colsum = new int[m + 1][n + 1];
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                rowsum[i][j] = rowsum[i][j - 1] + grid[i - 1][j - 1];
                colsum[i][j] = colsum[i - 1][j] + grid[i - 1][j - 1];
            }
        }
        for (int k = Math.min(m, n); k > 1; --k) {
            for (int i = 0; i + k - 1 < m; ++i) {
                for (int j = 0; j + k - 1 < n; ++j) {
                    int i2 = i + k - 1, j2 = j + k - 1;
                    if (check(grid, i, j, i2, j2)) {
                        return k;
                    }
                }
            }
        }
        return 1;
    }

    private boolean check(int[][] grid, int x1, int y1, int x2, int y2) {
        int val = rowsum[x1 + 1][y2 + 1] - rowsum[x1 + 1][y1];
        for (int i = x1 + 1; i <= x2; ++i) {
            if (rowsum[i + 1][y2 + 1] - rowsum[i + 1][y1] != val) {
                return false;
            }
        }
        for (int j = y1; j <= y2; ++j) {
            if (colsum[x2 + 1][j + 1] - colsum[x1][j + 1] != val) {
                return false;
            }
        }
        int s = 0;
        for (int i = x1, j = y1; i <= x2; ++i, ++j) {
            s += grid[i][j];
        }
        if (s != val) {
            return false;
        }
        s = 0;
        for (int i = x1, j = y2; i <= x2; ++i, --j) {
            s += grid[i][j];
        }
        if (s != val) {
            return false;
        }
        return true;
    }
}
~~~

<font color=#f00 size=6>**二维前缀和**</font>

### 304.二维区域和检索-矩阵不可变

![image-20240504195742866](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504195742866.png)



~~~java
class NumMatrix {
    private final int[][] sum;

    public NumMatrix(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        sum = new int[m + 1][n + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + matrix[i][j];
            }
        }
    }

    // 返回左上角在 (r1,c1) 右下角在 (r2,c2) 的子矩阵元素和
    public int sumRegion(int r1, int c1, int r2, int c2) {
        return sum[r2 + 1][c2 + 1] - sum[r2 + 1][c1] - sum[r1][c2 + 1] + sum[r1][c1];
    }
}
~~~

### 1314.矩阵区域和

![image-20240504201020353](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504201020353.png)

~~~java
class Solution {
    public int[][] matrixBlockSum(int[][] mat, int K) {
        int row = mat.length;
        int col = mat[0].length;
        int[][] res = new int[row][col];
        int[][] dp = new int[row + 1][col + 1];
        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= col; j++)
                dp[i][j] = mat[i - 1][j - 1] + dp[i][j - 1] + dp[i - 1][j] - dp[i - 1][j - 1];
        }
        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= col; j++) {
                int x0 = Math.max(i - K - 1, 0);
                int x1 = Math.min(i + K, row);
                int y0 = Math.max(j - K - 1, 0);
                int y1 = Math.min(j + K, col);
                res[i - 1][j - 1] = dp[x1][y1] - dp[x1][y0] - dp[x0][y1] + dp[x0][y0];
            }
        }
        return res;
    }
}
~~~

~~~java
class Solution {
    public int[][] matrixBlockSum(int[][] mat, int k) {
        int m = mat.length;
        int n = mat[0].length;
        int[][] sum = new int[m + 1][n + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + mat[i][j];
            }
        }
        int[][] ans = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int left = Math.max(j - k, 0);
                int right = Math.min(j + k, n - 1);
                int up = Math.max(i - k, 0);
                int down = Math.min(i + k, m - 1);
                ans[i][j] = sum[right + 1][down + 1] - sum[left][down + 1] - sum[right + 1][up] + sum[left][up];
            }
        }
        
        return ans;
    }
}
~~~

### 1277.统计全为1的正方形子矩阵

![image-20240504203023602](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504203023602.png)

<font color=#00f>**自己写出来的，但是代码的时间复杂度比较久**</font>

~~~java
class Solution {
    public int countSquares(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int[][] sum = new int[m + 1][n + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + mat[i][j];
            }
        }
        int ans = 0;
        for (int r = 0; r <= Math.min(n, m)-1; r++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    int right = j + r;
                    int down = i + r;
                    if (right > n - 1 || down > m - 1) {
                        continue;
                    }
                    int temp = sum[down + 1][right + 1] - sum[i][right + 1] - sum[down + 1][j] + sum[i][j];
                    
                    if (temp == (r+1) * (r+1)) {
                        ans++;
                    }
                }
            }
        }
        return ans;
    }
}
~~~

<font color=#f00>**通过动态规划做**</font>

~~~java
class Solution {
    public int countSquares(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int len = Math.min(m, n);
        boolean[][][] dp = new boolean[m][n][len];
        int count = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[i][j][0] = (matrix[i][j] == 1);
                count += dp[i][j][0] ? 1 : 0;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                for (int k = 1; k < len; k++) {
                    dp[i][j][k] = (matrix[i][j] == 1 && dp[i - 1][j][k - 1] && dp[i][j - 1][k - 1] && dp[i - 1][j - 1] [k - 1]);
                    if (dp[i][j][k]) {
                        count++;
                    }
                }
            }
        }
        return count;
    }

}
~~~

## 四十三.差分

<font color=#f00 size=6>**一维差分（扫描线）**</font>

### 1094.拼车

![image-20240504210657213](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504210657213.png)

~~~java
class Solution {
    public boolean carPooling(int[][] trips, int capacity) {
        int[] d = new int[1001];
        for (int[] t : trips) {
            int num = t[0], from = t[1], to = t[2];
            d[from] += num;
            d[to] -= num;
        }
        int s = 0;
        for (int v : d) {
            s += v;
            if (s > capacity) {
                return false;
            }
        }
        return true;
    }
}
~~~



### 2848.与车相交的点

![image-20240504211116107](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504211116107.png)

~~~java
class Solution {
    public int numberOfPoints(List<List<Integer>> nums) {
        // 定义差分数组
        int[] d = new int[102];
        for (List<Integer> num : nums) {
            int start = num.get(0);
            int end = num.get(1);
            d[start] += 1;
            d[end + 1] -= 1;
        }
        int ans = 0;
        int cunzai = 0;
        for (int i = 0; i <= 100; i++) {
            cunzai += d[i];
            if (cunzai > 0) {
                ans++;
            }
        }
        return ans;
    }
}
~~~

### 1893.检查是否区域内所有整数都被覆盖

![image-20240504211921600](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504211921600.png)

~~~java
class Solution {
    public boolean isCovered(int[][] ranges, int left, int right) {
        // 定义差分数组
        int[] d = new int[52];
        for (int[] num : ranges) {
            int start = num[0];
            int end = num[1];
            d[start] -= 1;
            d[end + 1] += 1;
        }
        int ans = 0;
        int cunzai = 0;
        for (int i = 0; i <= right; i++) {
            cunzai += d[i];
            if (cunzai == 0 && i >= left) {
                return false;
            }
        }
        return true;
    }
}
~~~

### 1109.航班预定统计

![image-20240504212843218](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504212843218.png)

~~~java
class Solution {
    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] d = new int[n+2];
        for(int[] booking:bookings){
            int first = booking[0];
            int last = booking[1];
            int seat = booking[2];
            d[first-1] += seat;
            d[last] -= seat;
        }
        int sum = 0;
        int[] ans = new int[n];
        for(int i = 0; i < n; i++){
            sum = sum + d[i];
            ans[i] = sum;
        }
        return ans;
    }
}
~~~

### 56.合并区间

![image-20240504213557276](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240504213557276.png)

~~~java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][2];
        }
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] interval1, int[] interval2) {
                return interval1[0] - interval2[0];
            }
        });
        List<int[]> merged = new ArrayList<int[]>();
        for (int i = 0; i < intervals.length; ++i) {
            int L = intervals[i][0], R = intervals[i][1];
            if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L) {
                merged.add(new int[]{L, R});
            } else {
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }
}
~~~

~~~java
class Solution {
    public int[][] merge(int[][] bookings) {
        int n = bookings.length;
        int[] d = new int[10005];
        for (int[] booking : bookings) {
            int first = booking[0];
            int last = booking[1];
            d[first - 1] += 1;
            d[last] -= 1;
        }
        int sum = 0;
        int start = 0;
        int end = 0;
        boolean flag = false;// 没找到第一个的时候设为false
        List<int[]> ans = new ArrayList<>();
        for (int i = 0; i < 10005; i++) {
            sum = sum + d[i];
            if (!flag && sum > 0) {
                start = i+1;
                end = i+1;
                flag = true;
            }
            if (flag && sum > 0) {
                end++;
            }
            if (flag && sum == 0) {
                ans.add(new int[] { start, end-1});
                flag = false;
            }

        }
        // 将结果转化为数组
        int[][] res = new int[ans.size()][2];
        ans.toArray(res);
        return res;
    }
}
~~~

### 57.插入区间

![image-20240505100159170](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505100159170.png)

~~~java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> ans = new ArrayList<>();
        int st = newInterval[0], ed = newInterval[1];
        boolean insert = false;
        for (int[] interval : intervals) {
            int s = interval[0], e = interval[1];
            if (ed < s) {
                if (!insert) {
                    ans.add(new int[] {st, ed});
                    insert = true;
                }
                ans.add(interval);
            } else if (e < st) {
                ans.add(interval);
            } else {
                st = Math.min(st, s);
                ed = Math.max(ed, e);
            }
        }
        if (!insert) {
            ans.add(new int[] {st, ed});
        }
        return ans.toArray(new int[ans.size()][]);
    }
}
~~~

<font color=#f00>**自己写的差分代码，代码有问题，会将不应该合并的位置进行合并**</font>按照差分数组左闭右开的写法

~~~java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        int n = intervals.length;
        int[] d = new int[10005];
        int first = newInterval[0];
        int last = newInterval[1];
        d[first] += 1;
        d[last] -= 1;
        for (int[] interval : intervals) {
            first = interval[0];
            last = interval[1];
            d[first] += 1;
            d[last] -= 1;
        }
        int sum = 0;
        int start = 0;
        int end = 0;
        boolean flag = false;// 没找到第一个的时候设为false
        List<int[]> ans = new ArrayList<>();
        for (int i = 0; i < 10005; i++) {
            sum = sum + d[i];
            if (!flag && sum > 0) {
                start = i + 1;
                end = i + 1;
                flag = true;
            }
            if (flag && sum > 0) {
                end++;
            }
            if (flag && sum == 0) {
                ans.add(new int[] { start - 1, end - 1 });
                flag = false;
            }

        }
        // 将结果转化为数组
        int[][] res = new int[ans.size()][2];
        ans.toArray(res);
        return res;
    }
}
~~~

### ==2406.==将区间分为最少组数

![image-20240505100447662](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505100447662.png)

<font color=#00f>**贪心算法+最小堆**</font>

~~~java
class Solution {
    public int minGroups(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        var pq = new PriorityQueue<Integer>();
        for (var p : intervals) {
            if (!pq.isEmpty() && pq.peek() < p[0]) pq.poll();
            pq.offer(p[1]);
        }
        return pq.size();
    }
}
~~~

<font color=#f00>**差分法**</font>

~~~java
class Solution {
    public int minGroups(int[][] intervals) {
        int[] d = new int[1000005];
        for (int[] interval : intervals) {
            int left = interval[0];
            int right = interval[1];
            d[left] += 1;
            d[right + 1] -= 1;
        }
        int max = 0;
        int sum = 0;
        for (int i = 1; i < 1000005; i++) {
            sum += d[i];
            max = Math.max(max, sum);
        }
        return max;
    }
}
~~~

<font color=#f00 size=6>**二维差分**</font>

### 2132.用邮票贴满网格图

![image-20240505145746146](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505145746146.png)

~~~java
class Solution {
    public boolean possibleToStamp(int[][] grid, int stampHeight, int stampWidth) {
        int m = grid.length;
        int n = grid[0].length;

        // 1. 计算 grid 的二维前缀和
        int[][] s = new int[m + 1][n + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                s[i + 1][j + 1] = s[i + 1][j] + s[i][j + 1] - s[i][j] + grid[i][j];
            }
        }

        // 2. 计算二维差分
        // 为方便第 3 步的计算，在 d 数组的最上面和最左边各加了一行（列），所以下标要 +1
        int[][] d = new int[m + 2][n + 2];
        for (int i2 = stampHeight; i2 <= m; i2++) {
            for (int j2 = stampWidth; j2 <= n; j2++) {
                int i1 = i2 - stampHeight + 1;
                int j1 = j2 - stampWidth + 1;
                if (s[i2][j2] - s[i2][j1 - 1] - s[i1 - 1][j2] + s[i1 - 1][j1 - 1] == 0) {
                    d[i1][j1]++;
                    d[i1][j2 + 1]--;
                    d[i2 + 1][j1]--;
                    d[i2 + 1][j2 + 1]++;
                }
            }
        }

        // 3. 还原二维差分矩阵对应的计数矩阵（原地计算）
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                d[i + 1][j + 1] += d[i + 1][j] + d[i][j + 1] - d[i][j];
                if (grid[i][j] == 0 && d[i + 1][j + 1] == 0) {
                    return false;
                }
            }
        }
        return true;
    }
}
~~~

### 2536.子矩阵元素加1

![image-20240505152448298](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505152448298.png)

~~~java
class Solution {
    public int[][] rangeAddQueries(int n, int[][] queries) {
        // 二维差分模板
        int[][] diff = new int[n + 2][n + 2], ans = new int[n][n];
        for (int[] q : queries) {
            int r1 = q[0], c1 = q[1], r2 = q[2], c2 = q[3];
            ++diff[r1 + 1][c1 + 1];
            --diff[r1 + 1][c2 + 2];
            --diff[r2 + 2][c1 + 1];
            ++diff[r2 + 2][c2 + 2];
        }
        // 用二维前缀和复原
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= n; ++j)
                ans[i - 1][j - 1] = diff[i][j] += diff[i][j - 1] + diff[i - 1][j] - diff[i - 1][j - 1];
        return ans;
    }
}
~~~

## 四十四.栈

<font color=#f00 size=6>**基础**</font>

### 1441.用栈操作数组（太简单了，没什么可复习的）

![image-20240505153343105](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505153343105.png)

~~~java
class Solution {
    public List<String> buildArray(int[] target, int n) {
        List<String> res = new ArrayList<>();
        int index = 0;
        for(int i = 1; i <= n; i++){
            if(i == target[index]){
                res.add("Push");
                index++;
            }else{
                res.add("Push");
                res.add("Pop");
            }
            if(index >= target.length){
                break;
            }
        }
        return res;
    }

}
~~~

### 1472.设计浏览器历史记录

![image-20240505155421378](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505155421378.png)

~~~java
class BrowserHistory {
    Deque<String> s1 = new LinkedList<>();
    Deque<String> s2 = new LinkedList();
    public BrowserHistory(String homepage) {
        s1.push(homepage);
    }
    
    public void visit(String url) {
        s1.push(url);
        s2.clear();
    }
    
    public String back(int steps) {
        for(int i = 0;i < steps && s1.size() > 1;i++){
            s2.push(s1.pop());
        }
        return s1.peek();
    }
    
    public String forward(int steps) {
        for(int i = 0;i < steps && !s2.isEmpty();i++){
            s1.push(s2.pop());
        }
        return s1.peek();
    }
}
~~~

<font color=#f00 size=6>**进阶**</font>

### 115.最小栈（简单）

![image-20240505160610440](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505160610440.png)

~~~java
class MinStack {
    private Stack<Integer> stack;
    private Stack<Integer> min_stack;
    public MinStack() {
        stack = new Stack<>();
        min_stack = new Stack<>();
    }
    public void push(int x) {
        stack.push(x);
        if(min_stack.isEmpty() || x <= min_stack.peek())
            min_stack.push(x);
    }
    public void pop() {
        if(stack.pop().equals(min_stack.peek()))
            min_stack.pop();
    }
    public int top() {
        return stack.peek();
    }
    public int getMin() {
        return min_stack.peek();
    }
}
~~~

### ==2434.==使用机器人打印字典序最小的字符串

![image-20240506100816949](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240506100816949.png)

~~~java
class Solution {
    public String robotWithString(String S) {
        StringBuilder ans = new StringBuilder();
        char[] s = S.toCharArray();
        int[] cnt = new int[26];
        for (int c : s) {
            cnt[c - 'a']++;
        }
        int min = 0;
        Deque<Character> st = new ArrayDeque<>();
        for (int i = 0; i < s.length; i++) {
            cnt[s[i]-'a']--;
            while(min<25 && cnt[min] == 0){
                min++;
            }
            st.push(s[i]);
            while(!st.isEmpty() && st.peek() - 'a' <= min){
                ans.append(st.poll());
            }
        }
        return ans.toString();
    }
}
~~~

### 716.最大栈

![image-20240506102603409](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240506102603409.png)

~~~java
class MaxStack {
    // 栈1用来存储入栈元素
    Deque<Integer> stack1;
    // 栈2用来存储每一层的最大值
    Deque<Integer> stack2;

    public MaxStack() {
        stack1 = new ArrayDeque<>();
        stack2 = new ArrayDeque<>();
    }
    
    public void push(int x) {
        stack1.offerLast(x);
        // 以下操作就是在stack2中每次都保存的是当前栈中所存元素的最大值
        if(stack2.isEmpty()){
            stack2.offerLast(x);
        }
        else{
            stack2.offerLast(Math.max(stack2.peekLast(),x));
        }
    }
    
    // stack1元素弹出的时候，相应的该层存储的最大元素也需要弹出
    public int pop() {
        stack2.pollLast();
        return stack1.pollLast();
    }
    
    // 直接返回stack1的栈顶元素即可
    public int top() {
        return stack1.peekLast();
    }
    
    // 同理，直接返回stack2的栈顶元素即可
    public int peekMax() {
        return stack2.peekLast();
    }
    
    // 弹出最大元素的操作稍微麻烦一点，因为我们弹出最大元素之后，还需要对stack2做一个更新
    public int popMax() {
        // 找到要弹出的元素的值
        int max = peekMax();
        // 暂时存储在最大元素上方的元素
        Deque<Integer> temp = new ArrayDeque<>();
        while(!stack1.isEmpty()){
            int cur = stack1.pollLast();
            stack2.pollLast();
            // 如果找到并弹出最大元素了，就退出循环
            if(cur == max){
                break;
            }
            // 否则表示这个值不应该被弹出，就先放入temp暂存
            temp.offerLast(cur);
        }
        // 再把temp中的元素放进stack1，同时更新stack2即可，使用上面写好的方法直接调用即可。
        while(! temp.isEmpty()){
            push(temp.pollLast());
        }
        return max;
    }
}

/**
 * Your MaxStack object will be instantiated and called as such:
 * MaxStack obj = new MaxStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.peekMax();
 * int param_5 = obj.popMax();
 */
~~~

<font color=#f00 size=6>**邻项消除**</font>

### 2696.删除子串后的字符串最小长度

![image-20240506105619539](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240506105619539.png)

~~~java
class Solution {
    public int minLength(String s) {
        StringBuilder ans = new StringBuilder();
        char[] S = s.toCharArray();
        Deque<Character> st = new ArrayDeque<>();
        for(char a:S){
            if(!st.isEmpty() && ((st.peekLast()=='A'&& a=='B')||(st.peekLast()=='C'&&a=='D'))){
                st.pollLast();
                continue;
            }
            st.offerLast(a);
        }
        while(!st.isEmpty()){
            ans.append(st.pollLast());
        }
        return ans.toString().length();
    }
}

~~~



### 1209.删除字符串中的所有相邻重复项II

![image-20240506110553139](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240506110553139.png)

~~~java
class Solution {
    public String removeDuplicates(String s, int k) {
        // 用两个栈，一个记录当前字符，一个记录一种字符的个数
        Deque<Character> st1 = new ArrayDeque<>();
        Deque<Integer> st2 = new ArrayDeque<>();
        char[] ss = s.toCharArray();
        for (char c : ss) {
            if (!st1.isEmpty() && !st2.isEmpty() && st1.peekLast() == c) {
                st1.offerLast(c);
                st2.offerLast(st2.pollLast() + 1);
                if (st2.peekLast() == k) {
                    st2.pollLast();
                    for (int i = 0; i < k; i++) {
                        st1.pollLast();
                    }
                }
            } else {
                st1.offerLast(c);
                st2.offerLast(1);
            }

        }
        StringBuilder ans = new StringBuilder();
        while (!st1.isEmpty()) {
            ans.append(st1.pollFirst());
        }
        return ans.toString();
    }
}
~~~

### ==2211.==统计道路上的碰撞次数(主要是思路难)

![image-20240506112232325](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240506112232325.png)

~~~java
class Solution {
    public int countCollisions(String s) {
        s = s.replaceAll("^L+", ""); // 前缀向左的车不会发生碰撞
        s = new StringBuilder(s).reverse().toString().replaceAll("^R+", ""); // 后缀向右的车不会发生碰撞
        return s.length() - (int) s.chars().filter(c -> c == 'S').count(); // 剩下非停止的车必然会碰撞
    }
}

~~~



<font color=#f00 size=6>**合法括号字符串**</font>

### 20.有效的括号

![image-20240506114542924](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240506114542924.png)

~~~java
public boolean isValid(String s) {
    Deque<Character> stack = new LinkedList<>();
    for (char i : s.toCharArray()) {
        if (i == '(') {
            stack.push(')');
        } else if (i == '[') {
            stack.push(']');
        } else if (i == '{') {
            stack.push('}');
        } else if (stack.isEmpty() || stack.pop() != i) {
            return false;
        }
    }
    return stack.isEmpty();
}
~~~

### ==856.==括号的分数

![image-20240506115342690](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240506115342690.png)

~~~java
class Solution {
    public int scoreOfParentheses(String s) {
        Deque<Integer> d = new ArrayDeque<>();
        d.addLast(0);
        for (char c : s.toCharArray()) {
            if (c == '(') d.addLast(0);
            else {
                int cur = d.pollLast();
                d.addLast(d.pollLast() + Math.max(cur * 2 , 1));
            }
        }
        return d.peekLast();
    }
}
~~~

### ==1006.==笨阶乘

![image-20240506153315490](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240506153315490.png)

~~~java
class Solution {
    public int clumsy(int n) {
        Deque<Integer> nums = new ArrayDeque<>();
        Deque<Character> ops = new ArrayDeque<>();
        // 维护运算符优先级
        Map<Character, Integer> map = new HashMap<>(){{
            put('*', 2);
            put('/', 2);
            put('+', 1);
            put('-', 1);
        }};
        char[] cs = new char[]{'*', '/', '+', '-'};
        for (int i = n, j = 0; i > 0; i--, j++) {
            char op = cs[j % 4];
            nums.addLast(i);
            // 如果「当前运算符优先级」不高于「栈顶运算符优先级」，说明栈内的可以算
            while (!ops.isEmpty() && map.get(ops.peekLast()) >= map.get(op)) {
                calc(nums, ops);
            }
            if (i != 1) ops.add(op);
        }
        // 如果栈内还有元素没有算完，继续算
        while (!ops.isEmpty()) calc(nums, ops);
        return nums.peekLast();
    }
    void calc(Deque<Integer> nums, Deque<Character> ops) {
        int b = nums.pollLast(), a = nums.pollLast();
        int op = ops.pollLast();
        int ans = 0;
        if (op == '+') ans = a + b;
        else if (op == '-') ans = a - b;
        else if (op == '*') ans = a * b;
        else if (op == '/') ans = a / b;
        nums.addLast(ans);
    }
}
~~~



### 2296.设计一个文本编译器

![image-20240506155331007](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240506155331007.png)

~~~java
class TextEditor {
    StringBuilder left = new StringBuilder(), right = new StringBuilder();

    public void addText(String text) {
        left.append(text);
    }

    public int deleteText(int k) {
        k = Math.min(k, left.length());
        left.setLength(left.length() - k);
        return k;
    }

    String text() {
        return left.substring(Math.max(left.length() - 10, 0));
    }

    public String cursorLeft(int k) {
        for (; k > 0 && !left.isEmpty(); --k) {
            right.append(left.charAt(left.length() - 1));
            left.deleteCharAt(left.length() - 1);
        }
        return text();
    }

    public String cursorRight(int k) {
        for (; k > 0 && !right.isEmpty(); --k) {
            left.append(right.charAt(right.length() - 1));
            right.deleteCharAt(right.length() - 1);
        }
        return text();
    }
}
~~~



## 四十五.队列



## 四十六.堆



## 四十七.字典树

### 208.实现Trie（前缀树）

![image-20240507152635753](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240507152635753.png)

~~~java
class Trie {
    class TrieNode {
        boolean end;
        TrieNode[] tns = new TrieNode[26];
    }

    TrieNode root;
    public Trie() {
        root = new TrieNode();
    }

    public void insert(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) p.tns[u] = new TrieNode();
            p = p.tns[u]; 
        }
        p.end = true;
    }

    public boolean search(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) return false;
            p = p.tns[u]; 
        }
        return p.end;
    }

    public boolean startsWith(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) return false;
            p = p.tns[u]; 
        }
        return true;
    }
}
~~~

















## 四十八.并查集

<font color=#00f>**标准模版：解决连通性问题**</font>

~~~java
int[] fa;

void init(int n) {
    fa = new int[n];
    for (int i = 0; i < n; i++) fa[i] = i;
}
//找到x的根节点，使用路径压缩了
int find(int x) {
        return x == fa[x] ? x : (fa[x] = find(fa[x]));
}
//合并两个节点
void union(int x, int y) {
    fa[find(x)] = find(y);
}
~~~



### 990.等式方程的可满足性

![image-20240507173130972](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240507173130972.png)

~~~java
class Solution {
    public boolean equationsPossible(String[] equations) {
        binchaji b = new binchaji();
        b.init(26);
        //将相等的归为一棵树
        for(String str:equations){
            if(str.charAt(1) == '='){
                b.union(str.charAt(0) -'a', str.charAt(3) - 'a');
            }
        }
        //检查不相等的情况是否是在不同的树上，如果是在相同的树就返回false
        for(String str:equations){
            if(str.charAt(1) == '!'){
                if(b.find(str.charAt(0) -'a') == b.find(str.charAt(3)-'a')){
                    return false;
                }
            }
        }
        return true;
    }
}

class binchaji {
    int[] fa;

    void init(int n) {
        fa = new int[n];
        for (int i = 0; i < n; i++)
            fa[i] = i;
    }

    // 找到x的根节点，使用路径压缩了
    int find(int x) {
        return x == fa[x] ? x : (fa[x] = find(fa[x]));
    }

    // 合并两个节点
    void union(int x, int y) {
        fa[find(x)] = find(y);
    }
}
~~~

### 737.句子相似性II

![image-20240507174019753](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240507174019753.png)

~~~java
class Solution {
    Map<String, String> map = new HashMap<>();
    public boolean areSentencesSimilarTwo(String[] words1, String[] words2, List<List<String>> pairs) {
        if(words1.length != words2.length)
            return false;
        for(List<String> pair: pairs)
            union(pair.get(0), pair.get(1));
        for(int i = 0; i < words1.length; i++)
        {
            if(!find(words1[i]).equals(find(words2[i])))
                return false;
        }
        return true;
    }
    
    public void union(String word1, String word2) {
        String x = find(word1);
        String y = find(word2);
        if(!x.equals(y))
            map.put(x, y);
    }
    
    public String find(String word) {
        while(map.containsKey(word) && map.get(word) != word)
            word = map.get(word);
        return word;
    }
}
~~~

<font color=#f00 size=6>**进阶**</font>

### 1202.交换字符串中的元素

![image-20240507180015244](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240507180015244.png)

~~~java
    public String smallestStringWithSwaps(String s, List<List<Integer>> pairs) {
        int length = s.length();
        //记录每个值的父节点值，初始状态默认每个值的父节点值都是他自己，祖宗节点也是他自己
        int[] parent = new int[length];
        for (int i = 0; i < length; i++)
            parent[i] = i;

        //pair中指向的两个值是可以自由交换的，所以他们是一阵营的，也就是祖宗是同一个。
        for (List<Integer> pair : pairs) {
            int ancestry0 = find(pair.get(0), parent);
            int ancestry1 = find(pair.get(1), parent);
            //ancestry0和ancestry1用哪一个成为他们的祖宗都是可以的
            parent[ancestry1] = ancestry0;
        }

        Map<Integer, Queue<Character>> map = new HashMap<>();
        for (int i = 0; i < length; i++) {
            //具有同一祖宗的，说明他们是一阵营的，把他们放到同一队列中
            int ancestry = find(i, parent);
            map.computeIfAbsent(ancestry, k -> new PriorityQueue<>()).offer(s.charAt(i));
        }

        //最后在进行合并
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < length; i++) {
            //找到i所在的队列，然后元素出队（这里的队列使用的是PriorityQueue，
            //其实就是个最小堆，每次出来的都是队列中最小的值）
            Queue queue = map.get(find(i, parent));
            stringBuilder.append(queue.poll());
        }
        return stringBuilder.toString();
    }

    //查找祖宗节点，只有当前的值等于他父节点值的时候才是祖宗节点。
    private int find(int i, int[] parent) {
        if (parent[i] != i) {
            //如果不是祖宗节点就继续往上查找
            parent[i] = find(parent[i], parent);
        }
        return parent[i];
    }
~~~















## 四十九.树状数组和线段树

<font color=#f00 size=6>**树状数组**</font>

<font color=#00f>**标准模版：适用于单点修改，进行区间查询、单点查询，可以再logn的复杂度求解**</font>

~~~java
int[] tree; //长度为n，初始化为0

int N;
int lowbit(int x) {return x&(-x);}

// 单点修改，第i个元素增加x
void update(int i, int x) {
    for (int p = i ; p < N ; p+=lowbit(p)) tree[p] += x;
}

// 查询前n项和
int query(int n) {
    int ans = 0;
    for (int p = n ; p>= 1; p -= lowbit(p)) ans += tree[p];
    return ans;
}
// 查询区间[a,b]的和
int query(int a, int b) {
    return query(b) - query(a-1);
}
~~~



### 307.区域和检索-数组可修改

![image-20240507153842606](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240507153842606.png)

~~~java
class NumArray {
    int[] tree;

    int N;

    int lowbit(int x) {
        return x & (-x);
    }

    // 查询前n项和
    int query(int n) {
        int ans = 0;
        for (int p = n; p >= 1; p -= lowbit(p))
            ans += tree[p];
        return ans;
    }

    // 查询区间[a,b]的和
    int query(int a, int b) {
        return query(b) - query(a - 1);
    }

    public NumArray(int[] nums) {

        this.N = nums.length;
        tree = new int[N + 1];
        for (int i = 0; i < N; i++) {
            update(i, nums[i]);
        }
    }

    public void update(int i, int x) {
        int old = query(i+1,i+1);
        for (int p = i+1; p <= N; p += lowbit(p))
            tree[p] += (x-old);
    }

    public int sumRange(int left, int right) {
        return query(left + 1, right + 1);
    }
}

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray obj = new NumArray(nums);
 * obj.update(index,val);
 * int param_2 = obj.sumRange(left,right);
 */
~~~

### ==3072.==将元素分配到两个数组中II

![image-20240507162907748](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240507162907748.png)

~~~java
class Fenwick {
    private final int[] tree;

    public Fenwick(int n) {
        tree = new int[n];
    }

    // 把下标为 i 的元素增加 1
    public void add(int i) {
        while (i < tree.length) {
            tree[i]++;
            i += i & -i;
        }
    }

    // 返回下标在 [1,i] 的元素之和
    public int pre(int i) {
        int res = 0;
        while (i > 0) {
            res += tree[i];
            i &= i - 1;
        }
        return res;
    }
}

class Solution {
    public int[] resultArray(int[] nums) {
        int[] sorted = nums.clone();
        Arrays.sort(sorted); // 只排序不去重
        int n = nums.length;

        List<Integer> a = new ArrayList<>(n); // 预分配空间
        List<Integer> b = new ArrayList<>();
        a.add(nums[0]);
        b.add(nums[1]);

        Fenwick t1 = new Fenwick(n + 1);
        Fenwick t2 = new Fenwick(n + 1);
        t1.add(Arrays.binarySearch(sorted, nums[0]) + 1);
        t2.add(Arrays.binarySearch(sorted, nums[1]) + 1);

        for (int i = 2; i < nums.length; i++) {
            int x = nums[i];
            int v = Arrays.binarySearch(sorted, x) + 1;
            int gc1 = a.size() - t1.pre(v); // greaterCount(a, v)
            int gc2 = b.size() - t2.pre(v); // greaterCount(b, v)
            if (gc1 > gc2 || gc1 == gc2 && a.size() <= b.size()) {
                a.add(x);
                t1.add(v);
            } else {
                b.add(x);
                t2.add(v);
            }
        }
        a.addAll(b);
        for (int i = 0; i < n; i++) {
            nums[i] = a.get(i);
        }
        return nums;
    }
}
~~~

~~~java

class Fenwick {
    private final int[] tree;
    private final int N;

    public Fenwick(int n) {
        tree = new int[n+1];
        this.N = n+1;
    }

    int lowbit(int x) {
        return x & (-x);
    }

    // 单点修改，第i个元素增加1
    void update(int i) {
        for (int p = i; p < N; p += lowbit(p))
            tree[p] += 1;
    }

    // 查询前n项和
    int query(int n) {
        int ans = 0;
        for (int p = n; p >= 1; p -= lowbit(p))
            ans += tree[p];
        return ans;
    }

    // 查询区间[a,b]的和
    int query(int a, int b) {
        return query(b) - query(a - 1);
    }
}

class Solution {
    public int[] resultArray(int[] nums) {
        int[] sorted = nums.clone();
        Arrays.sort(sorted);
        int n = nums.length;
        List<Integer> a = new ArrayList<>(n); // 预分配空间
        List<Integer> b = new ArrayList<>();
        a.add(nums[0]);
        b.add(nums[1]);

        Fenwick t1 = new Fenwick(n);
        Fenwick t2 = new Fenwick(n);
        t1.update(Arrays.binarySearch(sorted, nums[0])+1);
        t2.update(Arrays.binarySearch(sorted, nums[1])+1);
        for(int i = 2; i < n;i++){
            int x = nums[i];
            int v = Arrays.binarySearch(sorted, x) + 1;
            int gc1 = a.size() - t1.query(v);
            int gc2 = b.size() - t2.query(v);
            if (gc1 > gc2 || gc1 == gc2 && a.size() <= b.size()) {
                a.add(x);
                t1.update(v);
            } else {
                b.add(x);
                t2.update(v);
            }
        }


        a.addAll(b);
        for (int i = 0; i < n; i++) {
            nums[i] = a.get(i);
        }
        return nums;
    }
}
~~~









<font color=#f00 size=6>**线段树（无区间更新）**</font>



<font color=#f00 size=6>**Lazy线段树（有区间更新）**</font>



## 五十.离线算法









# 周赛

## 双周赛86|Gosper's Hack

### 2397.被列覆盖最多行数

![image-20240226093047263](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240226093047263.png)

~~~java
class Solution {
    public int maximumRows(int[][] mat, int numSelect) {
        int m = mat.length, n = mat[0].length;
        int[] mask = new int[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                mask[i] |= mat[i][j] << j;
            }
        }

        int ans = 0;
        for (int subset = 0; subset < (1 << n); subset++) {
            if (Integer.bitCount(subset) == numSelect) {
                int coveredRows = 0;
                for (int row : mask) {
                    if ((row & subset) == row) {
                        coveredRows++;
                    }
                }
                ans = Math.max(ans, coveredRows);
            }
        }
        return ans;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/maximum-rows-covered-by-columns/solutions/1798794/by-endlesscheng-dvxe/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



<font color=#f00>**Gosper's Hack**</font>

<font color=#00f>**Lowbit概念：计算机中计算一个数`x`的负数是二进制取反加一，`lowbit= -x & x`**</font>

~~~java
public class Solution {
    public int maximumRows(int[][] mat, int numSelect) {
        int m = mat.length, n = mat[0].length;
        int[] mask = new int[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                mask[i] |= mat[i][j] << j;
            }
        }

        int ans = 0;
        int subset = (1 << numSelect) - 1;
        while (subset < (1 << n)) {
            int coveredRows = 0;
            for (int row : mask) {
                if ((row & subset) == row) {
                    coveredRows++;
                }
            }
            ans = Math.max(ans, coveredRows);
            int lb = subset & -subset;
            int x = subset + lb;
            subset = ((subset ^ x) / lb >> 2) | x;
        }
        return ans;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/maximum-rows-covered-by-columns/solutions/1798794/by-endlesscheng-dvxe/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

[双周赛86视频教程](https://www.bilibili.com/video/BV1na41137jv/?vd_source=ed58d82293197f90b081de760b29a7f6)

## 双周赛91|动态规划 树上DFS

### 2466.统计构造好字符串的方案

~~~java
class Solution {
    public int countGoodStrings(int low, int high, int zero, int one) {
        //跟爬楼梯的做法一模一样，不过就想当与是多次爬楼梯的结果的集合，落到low和high区间都是解
        //定义f[i]表示长为i的字符串个数
        //f[i] = f[i-zero] + f[i-one]
        //f[0] = 1 因为当字符串为空的时候有一个解
        final int MOD = (int) 1e9 + 7;
        int ans = 0;
        var f = new int[high + 1]; // f[i] 表示构造长为 i 的字符串的方案数
        f[0] = 1; // 构造空串的方案数为 1
        for (int i = 1; i <= high; i++) {
            if (i >= one) f[i] = (f[i] + f[i - one]) % MOD;
            if (i >= zero) f[i] = (f[i] + f[i - zero]) % MOD;
            if (i >= low) ans = (ans + f[i]) % MOD;
        }
        return ans;
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/count-ways-to-build-good-strings/solutions/1964910/by-endlesscheng-4j22/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

## 第328场周赛

### 2535.数组元素和与数字和的绝对差

![image-20240315122314214](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240315122314214.png)

~~~java
class Solution {
    public int differenceOfSum(int[] nums) {
        int elementSum = 0, digitSum = 0;
        for (int num : nums) {
            elementSum += num;
            digitSum += getDigitSum(num);
        }
        return Math.abs(elementSum - digitSum);
    }

    public int getDigitSum(int num) {
        int sum = 0;
        while (num != 0) {
            sum += num % 10;
            num /= 10;
        }
        return sum;
    }
}
~~~

### 2536.子矩阵元素加1

<font color=#f00 size=6>**差分 前缀和**</font>

![image-20240315123843968](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240315123843968.png)

~~~java
~~~

### 2537.统计好子数组的数目

![image-20240315152452282](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240315152452282.png)

~~~java
~~~



### 1245.树的直径

![image-20240315114558803](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240315114558803.png)

~~~java
//dp思想最长的路径，可能是哪些情况
~~~



### 2538.最大价值与最小价值和的差值

![image-20240315114311490](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240315114311490.png)

~~~java
// 最小的价值和：一条路径只有一个节点
// 最大的价值和:路径的元素和越大越好，因为所有的元素都是正值，所以这个路径越长越好，所以一定是要找到根节点和叶子节点作为两个端点
// 开销：一条路径，去掉一个端点


class Solution {
    private List<Integer>[] g;
    private int[] price;
    private int ans;

    public long maxOutput(int n, int[][] edges, int[] price) {
        // 第一步建图
        this.price = price;
        g = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for(int[] e:edges){
            int x = e[0],y=e[1];
            g[x].add(y);
            g[y].add(x);
        }
        dfs(0,-1);
        return ans;
    }
    private int[] dfs(int x, int fa){
        //传递当前节点和父节点，返回带上端点的路径最大和，不带上端点的路径最大和
        int p = price[x];
        int max_s1 = price[x];
        int max_s2 = 0;
        for(int y:g[x]){
            //遍历当前节点所有的邻居，如果遍历到了当前的父节点，就跳过
            if(y == fa){
                continue;
            }
            int[] temp = dfs(y, x);
            int s1 = temp[0], s2 = temp[1];
            ans = Math.max(ans, Math.max(max_s1 + s2 , max_s2 + s1));
            max_s1 = Math.max(s1 + p, max_s1);
            max_s2 = Math.max(s2 + p, max_s2);//这里为什么要将p给补上，因为代码能执行到这里，必然不是叶子节点
        }
        return new int[]{max_s1, max_s2};
    }
}

~~~



## 第384场周赛



## 第385场周赛

### 3042.统计前后缀下标对I

![image-20240218155859760](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240218155859760.png)

~~~java
class Solution {
    public int countPrefixSuffixPairs(String[] words) {
        int n = words.length;
        int res = 0;
        for(int i = 0 ; i < n-1 ;i++){
            for(int j = i+1; j < n ;j++){
                if(isPrefixAndSuffix(words[i],words[j])){
                    res += 1;
                }
            }
        }
        return res;
    }
    private boolean isPrefixAndSuffix(String prefix, String suffix){
        if(suffix.startsWith(prefix) && suffix.endsWith(prefix)){
            return true;
        }
        return false;
    }
}
~~~



### 3044.出现频率最高的素数

<font color=#00f>**学会同时对于map的键值对增强循环方式利用entry**</font>

![image-20240218155710325](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240218155710325.png)

~~~java
class Solution {
    // 八个方向
    private static final int[][] DIRS = { { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 }, { -1, -1 }, { 0, -1 },
            { 1, -1 } };

    // 出现频率最高的素数
    public int mostFrequentPrime(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        Map<Integer, Integer> cnt = new HashMap<>();
        // 循环从每个格子作为起点
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // 每个格子进行枚举八个方向
                for (int[] d : DIRS) {
                    int x = i + d[0];
                    int y = j + d[1];
                    int v = mat[i][j];
                    // 判断是否走出了格子范围
                    while (x >= 0 && x < m && y >= 0 && y < n) {
                        v = v * 10 + mat[x][y];
                        if (isPrime(v)) {
                            //如果判定是素数，就将对应的素数数量加1
                            cnt.merge(v, 1, Integer::sum);
                        }
                        x += d[0];
                        y += d[1];
                    }
                }
            }
        } 
		//得到所有的情况后，记录到了cnt map中，之后遍历map获得所需的出现评率最高的素数
        int ans = -1;
        int maxCnt = 0;
        for (Map.Entry<Integer, Integer> e : cnt.entrySet()) {
            int v = e.getKey();
            int c = e.getValue();
            if (c > maxCnt) {
                ans = v;
                maxCnt = c;
            } else if (c == maxCnt) {
                ans = Math.max(ans, v);
            }
        }
        return ans;
    }

    // 判断是否是宿舍的方法
    private boolean isPrime(int n) {
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }
}

~~~

### 3043.最长公共前缀的长度

![image-20240218160048270](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240218160048270.png)

~~~java
class Solution {
    public int longestCommonPrefix(int[] arr1, int[] arr2) {
        int res = 0;
        for (int arri : arr1) {
            for (int arrj : arr2) {
                res = Math.max(res, longestCommonPrefixnum(arri, arrj));
            }
        }
        return res;

    }

    private int longestCommonPrefixnum(int num1, int num2) {
        String str1 = Integer.toString(num1);
        String str2 = Integer.toString(num2);
        int n1 = str1.length(), n2 = str2.length();
        int leng = (n1 < n2) ? n1 : n2;
        int ans = 0;
        for (int i = 0; i < leng; i++) {
            if(str1.charAt(i)==str2.charAt(i)){
                ans++;
            }else{
                break;
            }
        }
        return ans;
    }
}
~~~

~~~java
class Solution {
    public int longestCommonPrefix(int[] arr1, int[] arr2) {
        HashSet<String> set = new HashSet<>();
        for(int i:arr1){
            String s = String.valueOf(i);
            for(int j=1;j<=s.length();j++){
                set.add(s.substring(0,j));
            }
        }
        int ans = 0;
        for(int i:arr2){
            String s = String.valueOf(i);
            for(int j=1;j<=s.length();j++){
                if(set.contains(s.substring(0,j))){
                    ans = Math.max(ans, j);
                }else{
                    break;
                }
            }
        }
        return ans;
    }
}

作者：RUST_911
链接：https://leetcode.cn/problems/find-the-length-of-the-longest-common-prefix/solutions/2644375/javajian-dan-xie-fa-by-rust_911-jqd0/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~

<font color=#f00>**字典树方法高级**</font>

~~~java
class Solution {
    public int longestCommonPrefix(int[] arr1, int[] arr2) {
        Node node = new Node();
        Arrays.sort(arr2);
        int last = -1;
        for (int x : arr2) {
            if(x == last) continue;
            char[] cs = String.valueOf(x).toCharArray();
            Node t = node;
            for (char c : cs) {
                t = t.map.computeIfAbsent(c, e -> new Node());
            }
            last = x;
        }
        Arrays.sort(arr1);
        last = -1;
        int res = 0, cnt;
        for (int x : arr1) {
            if(x == last) continue;
            char[] cs = String.valueOf(x).toCharArray();
            Node t = node;
            cnt = 0;
            for (char c : cs) {
                t = t.map.get(c);
                if (t == null) break;
                cnt++;
            }
            res = Math.max(res, cnt);
            last = x;
        }
        return res;
    }
}

class Node {
    Map<Character, Node> map;
    Node() {
        map = new HashMap<>();
    }
}

作者：Redmos
链接：https://leetcode.cn/problems/find-the-length-of-the-longest-common-prefix/solutions/2644321/zi-dian-shu-xin-zhi-shi-dian-by-g31500-56kx/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



## 第391场周赛

### 3099.哈沙德数（ak，不用复习）

![image-20240420165519589](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240420165519589.png)

~~~java
class Solution {
    public int sumOfTheDigitsOfHarshadNumber(int x) {
        int su = sum(x);
        return (x / su) * su == x ? su : -1;
    }

    private int sum(int num) {
        List<Integer> nums = new ArrayList<>();
        while (num >= 10) {
            int temp = num % 10;
            num = num / 10;
            nums.add(temp);
        }
        nums.add(num);
        int ans = 0;
        for (int n : nums) {
            ans += n;
        }
        return ans;
    }
}
~~~

### 3100.换水问题II（ak，有点乱可以再看看）

![image-20240420170129379](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240420170129379.png)![image-20240420170138817](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240420170138817.png)

~~~java
class Solution {
    public int maxBottlesDrunk(int numBottles, int numExchange) {
        int BottlesDrunk = numBottles;
        int EmptyBottles = numBottles;
        while(numBottles != 0){
            numBottles =0;
            while(EmptyBottles >= numExchange){
                EmptyBottles -= numExchange;
                numExchange++;
                numBottles += 1;
            }
            BottlesDrunk += numBottles;
            EmptyBottles += numBottles;
        }
        return BottlesDrunk;
    }
}
~~~

### 3101.交替子数组计数（不用看）

![image-20240420171443681](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240420171443681.png)

~~~java
class Solution {
    public long countAlternatingSubarrays(int[] nums) {
        long ans = 0, cnt = 0;
        for (int i = 0; i < nums.length; i++) {
            //前半段判断是为了当i=0时i-1溢出，所以进行短路
            cnt = (i > 0 && nums[i] == nums[i - 1]) ? 1 : cnt + 1;
            ans += cnt; // 有 cnt 个以 i 为右端点的交替子数组
        }
        return ans;
    }
}

~~~

### ==3102.最小化曼哈顿距离==

![image-20240421163001078](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240421163001078.png)

![image-20240426143959393](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240426143959393.png)

<font color=#f00 size=5>**关键是旋转45度的坐标轴思想**</font>

![image-20240421164044474](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240421164044474.png)

~~~java
class Solution {
    public int minimumDistance(int[][] points) {
        TreeMap<Integer, Integer> xs = new TreeMap<>();
        TreeMap<Integer, Integer> ys = new TreeMap<>();
        for (int[] p : points) {
            xs.merge(p[0] + p[1], 1, Integer::sum);
            ys.merge(p[1] - p[0], 1, Integer::sum);
        }
        int ans = Integer.MAX_VALUE;
        for (int[] p : points) {
            int x = p[0] + p[1], y = p[1] - p[0];
            if (xs.get(x) == 1) xs.remove(x);
            else xs.merge(x, -1, Integer::sum);
            if (ys.get(y) == 1) ys.remove(y);
            else ys.merge(y, -1, Integer::sum);
            ans = Math.min(ans, Math.max(xs.lastKey() - xs.firstKey(), ys.lastKey() - ys.firstKey()));
            xs.merge(x, 1, Integer::sum);
            ys.merge(y, 1, Integer::sum);
        }
        return ans;
    }
}

~~~



## 第393场周赛

### 3114.替换字符可以得到的最晚时间

![image-20240420102958393](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240420102958393.png)

~~~java
class Solution {
    public String findLatestTime(String S) {
        char[] s = S.toCharArray();
        for (int h = 11; ; h--) {
            if (s[0] != '?' && s[0] - '0' != h / 10 || s[1] != '?' && s[1] - '0' != h % 10) {
                continue;
            }
            for (int m = 59; m >= 0; m--) {
                if (s[3] != '?' && s[3] - '0' != m / 10 || s[4] != '?' && s[4] - '0' != m % 10) {
                    continue;
                }
                return String.format("%02d:%02d", h, m);
            }
        }
    }
}
~~~

### 3115.质数的最大距离

![image-20240420103155807](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240420103155807.png)

判断一个数是不是质数，可以用这个数整除（2~根号（待判断的数））

~~~java
class Solution {
    public int maximumPrimeDifference(int[] nums) {
        int i = 0;
        while (!isPrime(nums[i])) {
            i++;
        }
        int j = nums.length - 1;
        while (!isPrime(nums[j])) {
            j--;
        }
        return j - i;
    }

    private boolean isPrime(int n) {
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return n >= 2;
    }
}
~~~

### ==3116.单面值组合的第K小金额==

![image-20240420104500147](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240420104500147.png)

<font color=#f00 size=6>**容斥原理**</font>

### ==3117.划分数组得到最小的值之和==

![image-20240420105307504](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240420105307504.png)

## 第394场周赛

### 3120.统计特殊字母的数量I

![image-20240421154906666](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240421154906666.png)

~~~java
class Solution {
    public int numberOfSpecialChars(String word) {
        boolean[] data = new boolean[128];
        for(char c : word.toCharArray()) {
            data[(int)c] = true;
        }
        int ans = 0;
        for(int i = 'a'; i <= 'z'; i++) {
            if(data[i] && data[i - 'a' + 'A']) {
                ans++;
            }
        }
        return ans;
    }
}
~~~

### 3121.统计特殊字母的数量II

![image-20240421154933400](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240421154933400.png)

~~~JAVA
class Solution {
    public int numberOfSpecialChars(String word) {
        int[] data = new int[128];
        Arrays.fill(data, -1);
        int index = 0;
        for(char c : word.toCharArray()) {
            if(c >= 'a' && c <= 'z') {
                data[c] = index;
            } else if(c >= 'A' && c <= 'Z' && data[c] == -1) {
                data[c] = index;
            }
            index++;
        }
        int ans = 0;
        for(int i = 'a'; i <= 'z'; i++) {
            if(data[i] != -1 && data[i] < data[i - 'a' + 'A']) {
                ans++;
            }
        }
        return ans;
    }
}

~~~

### ==3122.==使矩阵满足条件的最少操作次数

![image-20240421155231781](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240421155231781.png)

~~~java
class Solution {
    public int minimumOperations(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        //记录整个表中的所有的元素（0~9）的出现次数
        int[][] cnt = new int[n][10];
        for (int[] row : grid) {
            for (int j = 0; j < n; j++) {
                cnt[j][row[j]]++;
            }
        }
        //记忆化矩阵
        int[][] memo = new int[n][11];
        for (int[] row : memo) {
            Arrays.fill(row, -1); // -1 表示没有计算过
        }
        return m * n - dfs(n - 1, 10, cnt, memo);
    }
	//定义 dfs(i,j) 表示考虑前 i 列，且第 i+1 列都变成 j 时的最大保留不变元素个数。
    private int dfs(int i, int j, int[][] cnt, int[][] memo) {
        if (i < 0) {
            return 0;
        }
        if (memo[i][j] != -1) { // 之前计算过
            return memo[i][j];
        }
        int res = 0;
        for (int k = 0; k < 10; ++k) {
            if (k != j) {
                res = Math.max(res, dfs(i - 1, k, cnt, memo) + cnt[i][k]);
            }
        }
        return memo[i][j] = res; // 记忆化
    }
}

~~~

### ==100276.最短路径中的边==

![image-20240421161839421](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240421161839421.png)

~~~java
class Solution {
    public boolean[] findAnswer(int n, int[][] edges) {
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, i -> new ArrayList<>());
        for (int i = 0; i < edges.length; i++) {
            int[] e = edges[i];
            int x = e[0], y = e[1], w = e[2];
            g[x].add(new int[]{y, w, i});
            g[y].add(new int[]{x, w, i});
        }

        long[] dis = new long[n];
        Arrays.fill(dis, Long.MAX_VALUE);
        dis[0] = 0;
        PriorityQueue<long[]> pq = new PriorityQueue<>((a, b) -> Long.compare(a[0], b[0]));
        pq.offer(new long[]{0, 0});
        while (!pq.isEmpty()) {
            long[] dxPair = pq.poll();
            long dx = dxPair[0];
            int x = (int) dxPair[1];
            if (dx > dis[x]) {
                continue;
            }
            for (int[] t : g[x]) {
                int y = t[0];
                int w = t[1];
                long newDis = dx + w;
                if (newDis < dis[y]) {
                    dis[y] = newDis;
                    pq.offer(new long[]{newDis, y});
                }
            }
        }

        boolean[] ans = new boolean[edges.length];
        // 图不连通
        if (dis[n - 1] == Long.MAX_VALUE) {
            return ans;
        }

        // 从终点出发 BFS
        boolean[] vis = new boolean[n];
        dfs(n - 1, g, dis, ans, vis);
        return ans;
    }

    private void dfs(int y, List<int[]>[] g, long[] dis, boolean[] ans, boolean[] vis) {
        vis[y] = true;
        for (int[] t : g[y]) {
            int x = t[0];
            int w = t[1];
            int i = t[2];
            if (dis[x] + w != dis[y]) {
                continue;
            }
            ans[i] = true;
            if (!vis[x]) {
                dfs(x, g, dis, ans, vis);
            }
        }
    }
}

~~~



## 第396场周赛

### 100284.有效单词

![image-20240505152812617](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505152812617.png)

~~~java
class Solution {
    public boolean isValid(String word) {
        boolean v = false, w = false, d = false;
        int n = word.length();
        if (n < 3) return false;
        for (char c : word.toCharArray()) {
            if (!Character.isLetterOrDigit(c)) return false;
            if (Character.isLetter(c)) {
                c = Character.toLowerCase(c);
                if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') v = true;
                else w = true;
            }
        }
        return v && w;
    }
}

~~~

### 100275.K周期字符串需要的最少操作次数

![image-20240505152910919](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505152910919.png)

~~~java
class Solution {
    public int minimumOperationsToMakeKPeriodic(String word, int k) {
        //首先将word拆分成n/k个字符串
        int n = word.length();
        int len = n/k;
        Map<String, Integer> map = new HashMap<>();
        for(int i = 0; i < len;i++){
            String cur = word.substring(i*k,i*k+k);
            map.merge(cur,1,Integer::sum);
        }
        int max = 0;
        for(Map.Entry<String,Integer> entry:map.entrySet()){
            int temp = entry.getValue();
            max = Math.max(max,temp);
        }
        return len-max;
    }
}
~~~

### 3138.同位字符串连接的最小长度

![image-20240505153000553](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505153000553.png)

~~~java
class Solution {
    public int minAnagramLength(String s) {
        int n = s.length();
        int ans = n+1;
        for(int i = 1; i <= n; i++){
            boolean flag = true;
            if(n%i != 0){
                continue;
            }
            int len = n/i;
            String biaozhun = s.substring(0,i);
            for(int j = 1; j<len;j++){
                String cur = s.substring(j*i,j*i + i);
                if(!Pan(biaozhun,cur)){
                    flag = false;
                    break;
                }
            }
            if(flag){
                return i;
            }
        }
        return n;
    }
    
    //判断两个字符串是否是同位字符串
    private boolean Pan(String str1, String str2){
        int n = str1.length();
        int m = str2.length();
        if(n != m){
            return false;
        }
        char[] pan = new char[27];
        for(int i = 0; i< n;i++){
            char s1 = str1.charAt(i);
            char s2 = str2.charAt(i);
            pan[s1-'a' + 1]++;
            pan[s2-'a' + 1]--;
        }
        for(int i = 1;i < 27;i++){
            if(pan[i] != 0){
                return false;
            }
        }
        return true;
    }
}
~~~

### 100288.使数组中所有元素相等的最小开销

![image-20240505153053730](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240505153053730.png)

~~~java
class Solution {
    public int minCostToEqualizeArray(int[] nums, int cost1, int cost2) {
        int len = nums.length;
        Arrays.sort(nums);
        
        //正常写逻辑
        //使用前缀和
        int[] sum = new int[len+1];
        for(int i = 0; i< len;i++){
            sum[i+1] = sum[i] + nums[i];
        }
        int total = nums[len-1] * len;
        int queshao = total - sum[len];
        //判断特殊情况
        if(cost1 * 2 <= cost2){
            return queshao * cost1;
        }
        if(len == 2){
            return (nums[1]-nums[0]) * cost1;
        }
        int min = Integer.MAX_VALUE;
        for(int i = 1; i<len-1;i++){
            int left = nums[len-1] * i - sum[i];
            int right =nums[len-1] * (len-1 -i) -( sum[len-1] - sum[i]);
            if(left < right){
                min = Math.min(min,right - left);
            }else{
                min = Math.min(min,left - right);
            }
        }
        return min*cost1 + ((queshao - min) / 2) * cost2;
    }
}
~~~

~~~java
class Solution { 
    
    static final int MOD = (int) 1e9 + 7;

    public int minCostToEqualizeArray(int[] nums, int cost1, int cost2) {
        int n = nums.length;
        long max = 0, sum = 0, p = 0, x = 0;// p 最大可行操作二次数, x 剩余操作一次数
        for (int v : nums) {
            max = Math.max(max, v);
        }
        for (long v : nums) {
            v = max - v;// 需要增量
            sum += v;
            if (v <= x) {
                p += v;
                x -= v;
            } else {
                v -= x;
                long use = Math.min(v / 2, p);
                p += use + x;
                x = v - use * 2;
            }
        }
        if (cost1 * 2 <= cost2) {// 情况一
            return (int) (sum * cost1 % MOD);
        }
        long ans = p * cost2 % MOD;
        if (x > 0) {// 若 x == 0 等价于情况二，否则执行情况三的枚举
            long other = 0;
            long best = x * cost1;
            if (n > 2) {// 如果 n <= 1 则只能进行操作一
                do {
                    x++;
                    other += n - 1;
                    if (x >= other) {
                        best = Math.min(best, other * cost2 + (x - other) * cost1);
                    } else {
                        // 此时 x 第一次小于其他位置累计增加量
                        long t = x + other;
                        // 至多可行进的操作二的次数必然是 t / 2 向下取整
                        best = Math.min(best, (t / 2) * cost2 + (t % 2) * cost1);
                        if ((t & 1) != 0) {// 如果还需要一次操作一
                            t += n;// 则最大值需要的增加次数还需 + 1
                            best = Math.min(best, (t / 2) * cost2);
                        }
                    }
                } while (x >= other);// 若 x < other 结束枚举
            }
            ans += best;
            ans %= MOD;
        }
        return (int) ans;
    }
}
~~~









# 日常练习

### 14.最长公共前缀

![image-20240317195303052](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240317195303052.png)

<font color=#00f>**思路：将第一个字符串取出，之后遍历剩下的字符串判断每个位置的对应字符是否相同，如果是相同就继续往后遍历，如果不相同就截取**</font>

~~~java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        int length = strs[0].length();
        int count = strs.length;
        for (int i = 0; i < length; i++) {
            char c = strs[0].charAt(i);
            for (int j = 1; j < count; j++) {
                if (i == strs[j].length() || strs[j].charAt(i) != c) {
                    return strs[0].substring(0, i);
                }
            }
        }
        return strs[0];
    }
}
~~~

> [!IMPORTANT]
>
> - strs == null：这个条件用来检查传入的字符串数组本身是否存在，也就是说，它检查是否有对象被传递进来。如果strs为null，则表明没有传入有效的数组，试图访问这样的数组将会抛出NullPointerException。
> - strs.length == 0：这个条件则是检查即使strs不是null，它是否为空数组，即没有任何元素。在这种情况下，即使我们有了一个有效的数组对象，但是其内部不包含任何字符串元素。





### 55.跳跃游戏

![image-20240320170910335](Java重写：灵茶山艾府——基础算法精讲.assets/image-20240320170910335.png)

~~~java
class Solution {
    /**
     * 判断是否能够通过跳跃到达数组的每个元素
     * @param nums 给定的非负整数数组，每个元素代表在该位置可以跳跃的最大长度
     * @return 如果能够跳到数组的每个位置，则返回true；否则返回false
     */
    public boolean canJump(int[] nums) {
        // 动态规划：dp[i]表示从下标0到下标i是否能跳到i
        int n = nums.length;
        boolean[] dp = new boolean[n];
        dp[0] = true; // 起点可达
        // 遍历数组，更新dp数组
        for(int i = 1; i < n; i++){
            // 尝试从前面的每个位置跳到当前位置
            for (int j = 0; j < i; j++){
                // 如果之前的位置可达，并且从该位置可以跳到当前位置，则当前位置可达
                if(dp[j] && j + nums[j] >= i){
                    dp[i] = true;
                    break; // 找到一个可达路径即可退出内层循环
                }
            }
        }
        // 返回是否能跳到数组的最后一个位置
        return dp[n - 1];
    }
}

~~~



### 45.跳跃游戏II

![image-20240320172711447](Java重写：灵茶山艾府——基础算法精讲.assets/image-20240320172711447.png)

~~~java
class Solution {
    public int jump(int[] nums) {
        /**
         * 动态规划
         * dp[i] 表示到达下标 i 的最小跳跃次数,如果当前可以达到记录最小跳跃次数
         * 如果当前位置不能到达记录dp[i]=Integer.MAX_VALUE；
         */
        int n = nums.length;
        int[] dp = new int[n];
        dp[0] = 0;
        for (int i = 1; i < n; i++) {
            dp[i] = Integer.MAX_VALUE;
        }
        for (int i = 1; i < n; i++) {
            for(int j = 0; j < i; j++){
                if(dp[j] != Integer.MAX_VALUE && j + nums[j] >= i){
                    dp[i] = Math.min(dp[i], dp[j] + 1);
                }
            }
        }
        return dp[n - 1];
    }
}
~~~

### 274.H指数

![image-20240320183357800](Java重写：灵茶山艾府——基础算法精讲.assets/image-20240320183357800.png)

~~~java
class Solution {
    public int hIndex(int[] citations) {
        int n = citations.length;
        int[] count = new int[n + 1];
        for(int i = 0; i < n; i++){
            count[Math.min(n, citations[i])]++;
        }
        int s = 0;
        for (int i = n; i > 0; i--){
            s += count[i];
            if(s>= i){
                return i;
            }
        }
        return 0;
    }
}
~~~

### 275.H指数II

<font color=#00f>**这道题可以使用二分查找写**</font>

![image-20240320195150418](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240320195150418.png)

~~~java
class Solution {
    public int hIndex(int[] citations) {
        int n = citations.length;
        int max = 0;
        for (int i = n - 1; i >= 0; i--){
            if(citations[i] >= n - i){
                max = Math.max(max, n - i);
            }
        }
        return max;
    }
}
~~~

### ==380.==O（1）时间插入、删除和获取随机元素

![image-20240320194659832](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240320194659832.png)

~~~java
class RandomizedSet {
    //这个集合主要是为了更少时间复杂度进行随机返回集合中的一项的
    static int[] nums = new int[200010];
    Random random = new Random();
    Map<Integer, Integer> map = new HashMap<>();
    int idx = -1;

    public boolean insert(int val) {
        //如果当前元素的值已经存在了
        if (map.containsKey(val)) return false;
        nums[++idx] = val;
        map.put(val, idx);
        return true;
    }

    public boolean remove(int val) {
        if (!map.containsKey(val)) return false;
        int loc = map.remove(val);
        //如果当前移除的不是nums的最后一个元素，需要将最后一个元素删除，
        //将需要删除的元素用最后一个元素填充，并减小指针index位置
        if(loc != idx){
            map.put(nums[idx], loc);
        }
        nums[loc] = nums[idx--];
        return true;
    }

    public int getRandom() {
        return nums[random.nextInt(idx + 1)];
    }
}
~~~

### 238.除自身以外数组的乘积

![image-20240320200223974](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240320200223974.png)

~~~java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] s1 = new int[n + 10], s2 = new int[n + 10];
        s1[0] = s2[n + 1] = 1;
        for (int i = 1; i <= n; i++) s1[i] = s1[i - 1] * nums[i - 1];
        for (int i = n; i >= 1; i--) s2[i] = s2[i + 1] * nums[i - 1];
        int[] ans = new int[n];
        for (int i = 1; i <= n; i++) ans[i - 1] = s1[i - 1] * s2[i + 1];
        return ans;
    }
}

~~~

### ==134.==加油站

![image-20240320202928730](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240320202928730.png)

<font color=#f00>**看不懂贪心算法**</font>

~~~java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int curSum = 0;
        int totalSum = 0;
        int index = 0;
        for (int i = 0; i < gas.length; i++) {
            curSum += gas[i] - cost[i];
            totalSum += gas[i] - cost[i];
            if (curSum < 0) {
                index = (i + 1) % gas.length ; 
                curSum = 0;
            }
        }
        if (totalSum < 0) return -1;
        return index;
    }
}

~~~

~~~java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length;
        int totalgas = 0;
        for (int i = 0; i < n; i++) {
            int j = i;
            while (j < i + n) {
                totalgas += gas[j % n];
                if (totalgas <= cost[j % n]) {
                    break;
                }
                totalgas -= cost[j%n];
                j++;
                if (j == i + n) {
                    return i;
                }
            }
            totalgas = 0;
        }
        return -1;
    }
}
~~~

### 2007.从双倍数组中还原数组（每日一题）

![image-20240418184807070](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418184807070.png)

![image-20240418184931015](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240418184931015.png)

~~~java
class Solution {
    public int[] findOriginalArray(int[] changed) {
        Arrays.sort(changed);
        int[] ans = new int[changed.length / 2];
        int ansIdx = 0;
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int x : changed) {
            if (!cnt.containsKey(x)) { // x 不是双倍后的元素
                if (ansIdx == ans.length) {
                    return new int[0];
                }
                ans[ansIdx++] = x;
                cnt.merge(x * 2, 1, Integer::sum); // 标记一个双倍元素
            } else { // x 是双倍后的元素
                int c = cnt.merge(x, -1, Integer::sum); // 清除一个标记
                if (c == 0) {
                    cnt.remove(x);
                }
            }
        }
        return ans;
    }
}

~~~

<font color=#00f>**自己写的函数方法**</font>

~~~java
class Solution {
    public int[] findOriginalArray(int[] changed) {
        Arrays.sort(changed);
        int[] ans = new int[changed.length / 2];
        int ansIdx = 0;
        Map<Integer, Integer> cnt = new HashMap<>();

        for (int x : changed) {
            if (!cnt.containsKey(x)) {
                if (ansIdx == ans.length) {
                    return new int[0];
                }
                ans[ansIdx++] = x;
                cnt.compute(x * 2, (key, value) -> value == null ? 1 : value + 1);
            } else {
                cnt.compute(x, (key, value) -> value - 1);
                if (cnt.get(x) == 0) {
                    cnt.remove(x);
                }
            }
        }
        return ans;
    }
}
~~~



## MySQL:

### 183.从不订购的客户

![image-20240424122958686](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240424122958686.png)

![image-20240424123312920](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240424123312920.png)

~~~mysql
select a.Name as Customers
from Customers as a
left join Orders as b
on a.Id=b.CustomerId
where b.CustomerId is null;

~~~

~~~mysql
select *
from customers
where customers.id not in
(
    select customerid from orders
)

~~~



## ACM模式：

### 18.链表的基本操作

![image-20240405204930046](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405204930046.png)

~~~java
import java.util.*;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        //初始化链表
        LinkedList<Integer> list = new LinkedList<>();
        int n = sc.nextInt();
        for (int i=0;i<n;i++ ){
            int num = sc.nextInt();
            list.addFirst(num);// 从末尾开始添加元素
        }
        
        int m = sc.nextInt();
        for(int i = 0;i<m;i++){
            String operation  = sc.next();
            if (operation.equals("get")){
                int a = sc.nextInt();
                if(a>=1 && a<= list.size()){
                    System.out.println(list.get(a - 1));
                }else{
                    System.out.println("get fail");
                }
            }else if(operation.equals("delete")){
                int a = sc.nextInt();
                if(a>=1 && a<= list.size()){
                    list.remove(a - 1);
                    System.out.println("delete OK");
                }else{
                    System.out.println("delete fail");
                }
            }else if (operation.equals("insert")){
                int a = sc.nextInt();
                int e = sc.nextInt();
                if( a>=1 && a<= list.size()+1){//插入要+1
                    list.add(a - 1,e);
                    System.out.println("insert OK");
                }else{
                    System.out.println("insert fail");
                }
            }else if (operation.equals("show")){
                if(list.isEmpty()){
                    System.out.println("Link list is empty");
                }else{
                    for(int num : list){
                        System.out.print( num + " ");
                    }
                    System.out.println();
                }
            }   
        }
        sc.close();
    }
}
~~~



### ==21.==构件二叉树

![image-20240405204623353](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405204623353.png)

~~~java
import java.util.Scanner;
import java.util.HashMap;

public class Main{
    private static HashMap<Character,Integer> hashmap;
    private static int nodeIdx;
    public static class TreeNode{
        char val;
        TreeNode left;
        TreeNode right;
        public TreeNode(){};
        public TreeNode(char val){
            this.val = val;
        }
    }
    
    // 需要哈希表快速定位 中序序列中结点->下标的映射
    private static TreeNode buildTree(char[] preOrder,char[] inOrder){
        int n =preOrder.length;
        return build(preOrder,inOrder,0,n-1); // 最后两个参数控制inOrder的范围
    }
    
    private static TreeNode build(char[] preOrder,char[] inOrder,int l,int r){
        int n = preOrder.length;
        if(nodeIdx>=n || l>r) return null;
        char rootVal = preOrder[nodeIdx];
        TreeNode root = new TreeNode(rootVal);
        ++nodeIdx;
        int mid = hashmap.get(rootVal); 
        root.left = build(preOrder,inOrder,l,mid-1);
        root.right = build(preOrder,inOrder,mid+1,r);
        return root;
    }
    
    private static void dfs(TreeNode root){
        if(root==null)
            return;
        dfs(root.left);
        dfs(root.right);
        System.out.print(root.val);
    }
    
    
    public static void main(String[] args){
        Scanner in = new Scanner(System.in);
        while(in.hasNextLine()){
            String[] input = in.nextLine().split(" ");
            char[] preOrder = input[0].toCharArray();
            char[] inOrder = input[1].toCharArray();
            hashmap = new HashMap<Character,Integer>();
            nodeIdx = 0;
            for(int i=0;i<inOrder.length;i++){
                hashmap.put(inOrder[i],i);
            }
            TreeNode root = buildTree(preOrder,inOrder);
            dfs(root);
            System.out.println("");
        }
    }
}
~~~

### 22.二叉树的遍历

![image-20240405204655020](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405204655020.png)

~~~java
import java.util.*;
class TreeNode{
    char val;
    TreeNode left;
    TreeNode right;
    TreeNode(char val){
        this.val = val;
    }
}

public class Main{
    public static void main(String[] arg){
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        in.nextLine(); //忽略nextInt()的换行符。
        List<List<String>> list = new ArrayList<>();
        
    
        for(int i = 0; i < n; i++){
            String[] line = in.nextLine().split(" ");
            List<String> str = Arrays.asList(line);
            list.add(str);
        } 
        
        TreeNode root = builderListToTree(list);
        
        preOrderTree(root);
        System.out.println();
        inOrderTree(root);
        System.out.println();
        postOrderTree(root);
        System.out.println();
        
    }
    
    private static TreeNode builderListToTree(List<List<String>> list){
        if(list == null) return null;
        List<TreeNode> nodeList = new ArrayList<>();
        
        for(int i = 0; i<list.size(); i++){
            TreeNode node = new TreeNode(list.get(i).get(0).charAt(0));
            nodeList.add(node);
        }
    
        for(int i = 0; i< nodeList.size(); i++){
            int leftIndex = Integer.parseInt(list.get(i).get(1));
            int rightIndex = Integer.parseInt(list.get(i).get(2));
            if(leftIndex != 0){
                nodeList.get(i).left = nodeList.get(leftIndex-1);
            }
            if(rightIndex != 0){
                nodeList.get(i).right = nodeList.get(rightIndex-1);
            }
        }
        return nodeList.get(0);
    }
    
    private static void preOrderTree(TreeNode root){
        if(root == null) return;
        System.out.print(root.val);
        preOrderTree(root.left);
        preOrderTree(root.right);
    }
    
    private static void inOrderTree(TreeNode root){
        if(root == null) return;
        inOrderTree(root.left);
        System.out.print(root.val);
        inOrderTree(root.right);
    }
    
    private static void postOrderTree(TreeNode root){
        if(root == null) return;
        postOrderTree(root.left);
        postOrderTree(root.right);
        System.out.print(root.val);
    }
    
    
}
~~~

### 24.最长公共子序列

![image-20240405204744620](./Java重写：灵茶山艾府——基础算法精讲.assets/image-20240405204744620.png)

~~~java
import java.util.*;
public class Main{
    private static int[][] cache;
    private static char[] c1;
    private static char[] c2;
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        while(sc.hasNext()){
            String[] s = sc.nextLine().split(" ");
            c1 = s[0].toCharArray();
            c2 = s[1].toCharArray();
            cache = new int[c1.length][c2.length];
            for(int i = 0; i < c1.length;i++){
                Arrays.fill(cache[i],-1);
            }
            System.out.println(dfs(c1.length-1,c2.length-1));
        }
        sc.close();
    }
    
    
    public static int dfs(int i, int j) {
        if(i < 0 || j < 0) {
            return 0;
        }
        
        if(cache[i][j] != -1) {
            return cache[i][j];
        }
        if(c1[i] == c2[j]) {
            return cache[i][j] = dfs(i-1,j-1)+1;
        }  
        return cache[i][j] = Math.max(dfs(i-1,j),dfs(i,j-1));
        
    }
    
    
}
~~~

~~~java
import java.util.*;

public class Main{
    public static void main (String[] args) {
        Scanner sc=new Scanner(System.in);
        while(sc.hasNext()){
            char[] stringa = sc.next().toCharArray();
            char[] stringb = sc.next().toCharArray();
            int lengtha =stringa.length;
            int lengthb =stringb.length;
            int[][] res = new int[lengtha+1][lengthb+1];
            for (int i=0;i<lengtha+1 ;i++ ){
                res[i][0]=0;
            } 
            for (int j=0;j<lengthb+1 ;j++ ){
                res[0][j]=0;
            }
            for(int i=1;i<lengtha+1;i++){
                for(int j=1;j<lengthb+1;j++){
                    if(stringa[i-1]==stringb[j-1]){
                        res[i][j]=res[i-1][j-1]+1;
                    }else{
                        res[i][j]=Math.max(res[i-1][j],res[i][j-1]);
                    }
                }                
            }
            System.out.println(res[lengtha][lengthb]);            
        }
    }
    
}
~~~

