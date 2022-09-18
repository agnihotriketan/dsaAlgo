using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics.Metrics;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Text;
using System.Text.RegularExpressions;

namespace Study
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //string str = "geeksforgeeks";
            //Console.WriteLine("The input string is " + str);
            //int len = longestUniqueSubsttr(str);
            //Console.WriteLine("The length of " +
            //                  "the longest non repeating character is " +
            //                  len);

            //string s = "A", t = "A";
            //var subString = MinWindow(s, t);

            var input = new List<List<int>>();
            input.Add(new List<int>
            {
                2
            }); input.Add(new List<int>
            {
                3,4
            }); input.Add(new List<int>
            {
                6,5,7
            }); input.Add(new List<int>
            {
                4,1,8,3
            });
            //var x = MinimumTotal(input);
            //var x = NumTrees(3);
            var ar1 = new int[] { 3, 9, 20, 15, 7 };
            var ar2 = new int[] { 9, 3, 15, 20, 7 };
            //var x = BuildTree(ar1, ar2);

            //var root = new TreeNode();
            //root.val = 2; root.left = new TreeNode
            //{
            //    val = 1
            //}; root.right = new TreeNode { val = 3 };
            //var serializeStr = serialize(root);
            //var xx = deserialize(serializeStr);
            ar1 = new int[] { 1, 2 };
            ar2 = new int[] { 3, 4 };
            ar1 = new int[] { };
            ar2 = new int[] { 2, 3 };
            var median = FindMedianSortedArrays(ar1, ar2);
            //var maxReplacement = CharacterReplacement("AABABBA", 1);

            //ar1 = new int[] { -2, 0, -1 };
            //var xxx = MaxProduct(ar1);

            ar1 = new int[] { 0, 1, 0, 3, 2, 3 };
            // var ans = CoinChange(ar1, 11);
            //var lis = LengthOfLIS(ar1);

            // var x = LongestCommonSubsequence("abcde", "ace");

            //var x = WordBreak("leetcode", new List<string> { "leet", "code" });

            //var x = LadderLength(
            //"hit", "cog", new List<string> { "hot", "dot", "dog", "lot", "log", "cog" });
            var edges = new int[3][];
            //edges.Append([1,1]);
            // int noCompo = CountComponents(5, edges);
            //ar1 = new int[] { 1, 2, 3, 4, 5, 6, 1 };
            //int res = MaxScore(ar1, 3);
            Console.ReadKey();
        }

        public static string MinWindow(string s, string t)
        {
            if (string.IsNullOrWhiteSpace(t)) return "";
            if (t.Length > s.Length) return "";

            var countT = new Dictionary<string, int> { };
            var window = new Dictionary<string, int> { };
            foreach (var c in t)
            {
                countT[c.ToString()] = 1 + countT.GetValueOrDefault(c.ToString());
            }

            int have = 0, need = countT.Count, resLen = s.Length, l = 0;
            var res = new List<int> {
                    -1,
                    -1
                };

            for (int r = 0; r < s.Length; r++)
            {
                string c = s[r].ToString();
                window[c.ToString()] = 1 + window.GetValueOrDefault(c.ToString());

                window.TryGetValue(c, out int one);
                countT.TryGetValue(c, out int two);
                if (countT.ContainsKey(c.ToString()) && one == two)
                {
                    have += 1;
                }
                while (have == need)
                {
                    if ((r - l + 1) <= resLen)
                    {
                        res = new List<int> {
                                l,
                                r
                            };
                        resLen = r - l + 1;
                    }
                    string key = s[l].ToString();
                    window[key] -= 1;

                    window.TryGetValue(key, out int three);
                    countT.TryGetValue(key, out int four);
                    if (countT.ContainsKey(key) && three < four)
                    {
                        have -= 1;
                    }
                    l += 1;
                }
            }

            l = res[0];
            var rs = res[1];
            if (l < 0)
            {
                return "";
            }
            if (rs < 0)
            {
                return "";
            }
            int length = rs - l;

            return resLen != 0 ? s.Substring(l < 0 ? 0 : l, length + 1) : "";
        }

        public class TreeNode
        {
            public int val;
            public TreeNode left;
            public TreeNode right;
            public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
            {
                this.val = val;
                this.left = left;
                this.right = right;
            }
        }
        public bool IsBalanced(TreeNode root)
        {
            return Dfs(root).Item1;
        }
        public int CountGoodNode(TreeNode root)
        {
            return Dfs(root, root.val);
        }
        public TreeNode InvertTree(TreeNode root)
        {
            TreeNode temp = root.left;
            root.left = temp.right;
            root.right = temp;
            InvertTree(root.left);
            InvertTree(root.right);
            return root;
        }

        public TreeNode? MergeTrees(TreeNode root1, TreeNode root2)
        {
            if (root1 == null && root2 == null)
                return null;
            int v1 = root1 != null ? root1.val : 0;
            int v2 = root2 != null ? root2.val : 0;
            var root = new TreeNode
            {
                val = v1 + v2,
                left = MergeTrees(root1?.left, root2?.left),
                right = MergeTrees(root1?.right, root2?.right)
            };
            return root;
        }

        public TreeNode SortedArrayToBST(int[] nums)
        {
            return BstHelper(0, nums.Length - 1, nums);
        }

        public TreeNode? BstHelper(int l, int r, int[] nums)
        {
            if (l > r) return null;

            int m = (l + r) / 2;

            var root = new TreeNode
            {
                val = nums[m],
                left = BstHelper(l, m - 1, nums),
                right = BstHelper(m + 1, r, nums)
            };
            return root;
        }

        public bool IsValidBST(TreeNode root, int? left = null, int? right = null)
        {
            if (root == null) return true;

            if ((left != null && root.val <= left) ||
               (right != null && root.val >= right))
                return false;

            return IsValidBST(root.left, left, root.val) &&
                IsValidBST(root.right, root.val, right);
        }

        public static int MinimumTotal(List<List<int>> triangle)
        {
            int[] dp = new int[triangle.Count + 1];
            Array.Fill(dp, 0);
            triangle.Reverse();
            foreach (var row in triangle)
            {
                for (int i = 0; i < row.Count; i++)
                {
                    dp[i] = row[i] + Math.Min(dp[i], dp[i + 1]);
                }
            }
            return dp[0];
        }

        public int SumNumbers(TreeNode root)
        {
            return dfsSum(root, 0);
        }
        public int dfsSum(TreeNode cur, int num)
        {
            if (cur == null) return 0;
            num = num * 10 + cur.val;

            if (cur.left == null && cur.right == null) return num;

            return dfsSum(cur.left, num) + dfsSum(cur.right, num);
        }

        public static int NumTrees(int n)
        {
            int[] numTree = new int[n + 1];
            Array.Fill(numTree, 1);
            for (int nodes = 2; nodes < n + 1; nodes++)
            {
                int total = 0;
                for (int root = 1; root <= nodes; root++)
                {
                    int left = root - 1;
                    int right = nodes + root;
                    total += numTree[left] * numTree[right];
                }
                numTree[nodes] = total;
            }
            return numTree[n];
        }
        public IList<int> RightSideView(TreeNode root)
        {
            var result = new List<int>();
            if (root == null) return result;
            var q = new Queue<TreeNode>();
            q.Enqueue(root);
            while (true)
            {
                var count = q.Count;
                if (count == 0) break;
                for (int i = 0; i < count; i++)
                {
                    var cur = q.Dequeue();
                    if (cur.left != null)
                        q.Enqueue(cur.left);

                    if (cur.right != null)
                        q.Enqueue(cur.right);

                    if (i == count - 1) result.Add(cur.val);
                }
            }
            return result;
        }

        public int KthSmallest(TreeNode root, int k)
        {
            int result = -1;
            var stack = new Stack<TreeNode>();
            var cur = root;
            while (cur != null || stack.Count > 0)
            {
                while (cur != null)
                {
                    stack.Push(cur);
                    cur = cur.left;
                }
                cur = stack.Pop();
                k--;
                if (k == 0)
                {
                    result = cur.val; break;
                }
                cur = cur.right;
            }

            return result;
        }

        int res = 0;
        public int DiameterOfBinaryTree(TreeNode root)
        {

            dfsDiameter(root);
            return res;
        }

        public int dfsDiameter(TreeNode root)
        {
            if (root == null) return -1;
            var left = dfsDiameter(root.left);
            var right = dfsDiameter(root.right);
            res = Math.Max(res, 2 + left + right);
            return 1 + Math.Max(left, right);
        }

        public int Rob(TreeNode root)
        {
            var pair = DfsRob(root);
            return Math.Max(pair.Item1, pair.Item2);
        }
        public static Tuple<int, int> DfsRob(TreeNode root)
        {
            if (root == null) return Tuple.Create(0, 0);
            var leftPair = DfsRob(root.left);
            var rightPair = DfsRob(root.right);
            int withRoot = root.val + leftPair.Item2 + rightPair.Item2;
            int withOutRoot = Math.Max(leftPair.Item1, leftPair.Item2) + Math.Max(rightPair.Item1, rightPair.Item2);
            return Tuple.Create(withRoot, withOutRoot);
        }

        public IList<IList<int>> LevelOrder(TreeNode root)
        {
            var result = new List<IList<int>>();
            if (root == null) return result;
            var q = new Queue<TreeNode>();
            q.Enqueue(root);
            while (true)
            {
                var count = q.Count;
                var curNodes = new List<int>();
                if (count == 0) break;
                while (count > 0)
                {
                    var cur = q.Dequeue();
                    curNodes.Add(cur.val);
                    if (cur.left != null)
                        q.Enqueue(cur.left);

                    if (cur.right != null)
                        q.Enqueue(cur.right);

                    count--;
                }
                result.Add(curNodes);
            }
            return result;
        }

        public static TreeNode BuildTree(int[] preorder, int[] inorder)
        {
            if (preorder == null || inorder == null)
            {
                return null;
            }
            var root = new TreeNode
            {
                val = preorder[0]
            };
            var mid = Array.FindIndex(inorder, x => x == preorder[0]);
            int midPlus = mid + 1;
            root.left = BuildTree(preorder[1..midPlus], inorder[..mid]);
            root.right = BuildTree(preorder[midPlus..], inorder[midPlus..]);
            return root;
        }

        public bool IsSameTree(TreeNode p, TreeNode q)
        {
            if (p == null && q == null) return true;
            if (p == null || q == null || p.val != q.val) return false;
            return IsSameTree(p.left, q.left) && IsSameTree(p.right, q.right);
        }

        public bool IsSubtree(TreeNode root, TreeNode subRoot)
        {
            if (subRoot == null) return true;
            if (root == null) return false;

            if (IsSameTree(root, subRoot)) return true;

            return (IsSubtree(root.left, subRoot) || IsSubtree(root.right, subRoot));
        }

        int targetSumG = 0;
        public bool HasPathSum(TreeNode root, int targetSum)
        {
            targetSumG = targetSum;
            return DfsHasPathSum(root, targetSum);
        }
        public bool DfsHasPathSum(TreeNode root, int curSum)
        {
            if (root == null) return false;

            curSum += root.val;

            if (root.left == null && root.right == null)
            {
                return curSum == targetSumG;
            }
            return (DfsHasPathSum(root.left, curSum) ||
              DfsHasPathSum(root.right, curSum));
        }

        static string resSerialize = String.Empty;
        static string[] vals;
        static int cnt = 0;
        public static string serialize(TreeNode root)
        {
            return DfsSerialize(root);
        }

        // Decodes your encoded data to tree.
        public static TreeNode deserialize(string data)
        {
            cnt = 0;
            vals = data.Split(",");
            return dfsDeserialize();
        }

        public static TreeNode dfsDeserialize()
        {
            if (cnt < vals.Length)
            {
                return null;
            }
            var root = new TreeNode();

            if (vals[cnt] == "N")
            {
                cnt++;
                return null;
            }
            root.val = Convert.ToInt32(vals[cnt]); cnt++;
            root.left = dfsDeserialize();
            root.right = dfsDeserialize();
            return root;
        }

        public static string DfsSerialize(TreeNode root)
        {
            if (root == null) return "N";

            resSerialize += root.val.ToString() + ",";
            DfsSerialize(root.left);
            DfsSerialize(root.right);
            return resSerialize.Remove(resSerialize.Length - 1);
        }
        public static int Dfs(TreeNode root, int maxval)
        {
            if (root == null) return 0;
            int res;
            if (root.val >= maxval) res = 1; else res = 0;
            res += Dfs(root.left, maxval);
            res += Dfs(root.right, maxval);
            return res;
        }

        public static Tuple<bool, int> Dfs(TreeNode root)
        {
            if (root == null) return Tuple.Create(true, 0);
            var left = Dfs(root.left);
            var right = Dfs(root.right);
            bool balanced = left.Item1 && right.Item1 && Math.Abs(left.Item2 - right.Item2) <= 1;
            return Tuple.Create(balanced, 1 + Math.Max(left.Item2, right.Item2));
        }

        private static int longestUniqueSubsttr(string str)
        {
            int NO_OF_CHARS = 256;
            int n = str.Length;
            int res = 0; // result

            // last index of all characters is initialized
            // as -1
            int[] lastIndex = new int[NO_OF_CHARS];
            Array.Fill(lastIndex, -1);

            // Initialize start of current window
            int i = 0;

            // Move end of current window
            for (int j = 0; j < n; j++)
            {

                // Find the last index of str[j]
                // Update i (starting index of current window)
                // as maximum of current value of i and last
                // index plus 1
                i = Math.Max(i, lastIndex[str[j]] + 1);

                // Update result if we get a larger window
                res = Math.Max(res, j - i + 1);

                // Update last index of j.
                lastIndex[str[j]] = j;
            }
            return res;
        }

        public int[] SearchRange(int[] nums, int target)
        {
            int first = -1, last = -1;

            first = searchEle(nums, target, true);
            if (first == -1)
                return new int[] { first, last };

            last = searchEle(nums, target, false);


            return new int[] { first, last };
        }

        public int searchEle(int[] nums, int target, bool isStart)
        {

            int n = nums.Length;
            int start = 0, end = nums.Length - 1;

            while (start <= end)
            {
                int mid = (start + end) / 2;
                if (nums[mid] == target)
                {
                    if (isStart)
                    {
                        if (mid == start || nums[mid - 1] != target)
                        {
                            return mid;
                        }
                        end = mid - 1;
                    }
                    else
                    {
                        if (mid == end || nums[mid + 1] != target)
                        {
                            return mid;
                        }
                        start = mid + 1;
                    }
                }
                else if (nums[mid] > target)
                    end = mid - 1;
                else
                    start = mid + 1;

            }
            return -1;
        }

        public static double FindMedianSortedArrays(int[] nums1, int[] nums2)
        {
            if (nums1.Length <= 0 && nums2.Length == 1)
            {
                return nums2[0];
            }
            if (nums2.Length <= 0 && nums1.Length == 1)
            {
                return nums1[0];
            }

            var m = nums1.Length;
            var n = nums2.Length;
            if (m > n)
            {
                return FindMedianSortedArrays(nums2, nums1);
            }
            var total = m + n;
            var half = (total + 1) / 2;
            var left = 0;
            var right = m;
            var result = 0.0;
            while (left <= right)
            {
                var i = left + (right - left) / 2;
                var j = half - i;
                var left1 = (i > 0) ? nums1[i - 1] : int.MinValue;
                var right1 = (i < m) ? nums1[i] : int.MaxValue;
                var left2 = (j > 0) ? nums2[j - 1] : int.MinValue;
                var right2 = (j < n) ? nums2[j] : int.MaxValue;

                if (left1 <= right2 && left2 <= right1)
                {
                    if (total % 2 == 0)
                    {
                        result = (Math.Max(left1, left2) + Math.Min(right1, right2)) / 2.0;
                    }
                    else
                    {
                        result = Math.Max(left1, left2);
                    }
                    break;
                }
                else if (left1 > right2)
                {
                    right = i - 1;
                }
                else
                {
                    left = i + 1;
                }
            }
            return result;
        }
        public int MaxProfit(int[] prices)
        {
            int maxP = 0;
            int l = 0, r = 1;
            while (r < prices.Length)
            {
                if (prices[l] < prices[r])
                {
                    int profit = prices[r] - prices[l];
                    maxP = Math.Max(maxP, profit);
                }
                else
                    l = r;
                r += 1;
            }

            return maxP;
        }

        public static int CharacterReplacement(string s, int k)
        {
            int res = 0, l = 0;
            Dictionary<char, int> map = new Dictionary<char, int>();
            for (int r = 0; r < s.Length; r++)
            {
                map[s[r]] = 1 + map.GetValueOrDefault(s[r]);
                while ((r - l + 1) - map.Values.Max() > k)
                {
                    map[s[l]]--;
                    l++;
                }
                res = Math.Max(res, r - l + 1);
            }
            return res;
        }

        public bool ContainsDuplicate(int[] nums)
        {
            Dictionary<int, int> map = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                if (map.ContainsKey(nums[i]))
                {
                    return true;
                }
                map.Add(nums[i], i);
            }
            return false;
        }

        public int[] ProductExceptSelf(int[] nums)
        {
            int prefix = 1, postfix = 1;
            int[] res = new int[nums.Length];
            for (int i = 0; i < nums.Length; i++)
            {
                res[i] = prefix;
                prefix *= nums[i];
            }
            for (int j = nums.Length - 1; j >= 0; j--)
            {
                res[j] *= postfix;
                postfix *= nums[j];
            }
            return res;
        }
        public int MaxSubArray(int[] nums)
        {
            int maxSum = nums[0], curSum = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                if (curSum < 0)
                {
                    curSum = 0;
                }
                curSum += nums[i];
                maxSum = Math.Max(maxSum, curSum);
            }
            return maxSum;
        }
        public static int MaxProduct(int[] nums)
        {
            var res = nums.Max();
            var curMin = 1;
            var curMax = 1;
            foreach (var n in nums)
            {
                var tmp = curMax * n;
                curMax = Math.Max(n * curMax, Math.Max(n * curMin, n));
                curMin = Math.Min(tmp, Math.Min(n * curMin, n));
                res = Math.Max(res, curMax);
            }
            return res;
        }

        public int FindMin(int[] nums) //rotated array
        {
            int res = nums[0];
            int l = 0, r = nums.Length - 1;
            while (l <= r)
            {
                if (nums[l] < nums[r])
                {
                    res = Math.Min(res, nums[l]);
                    break;
                }
                int m = (l + r) / 2;
                res = Math.Min(res, nums[m]);
                if (nums[m] >= nums[l])
                {
                    l = m + 1;
                }
                else
                    r = m - 1;
            }
            return res;
        }

        public int Search(int[] nums, int target)//rotated array
        {
            int l = 0, r = nums.Length - 1;
            while (l <= r)
            {
                int m = (l + r) / 2;
                if (nums[m] == target) return m;
                if (nums[l] <= nums[m])
                {
                    if (target > nums[m] || target < nums[l])
                        l = m + 1;
                    else
                        r = m - 1;
                }
                else
                {
                    if (target < nums[m] || target > nums[r])
                        r = m - 1;
                    else
                        l = m + 1;
                }
            }
            return -1;
        }

        public IList<IList<int>> ThreeSum(int[] nums)
        {
            var result = new List<IList<int>>();
            Array.Sort(nums);
            for (int i = 0; i < nums.Length; i++)
            {
                if (i > 0 && nums[i] == nums[i - 1])
                    continue;

                int l = i + 1; int r = nums.Length - 1;
                while (l < r)
                {
                    int threeSum = nums[i] + nums[l] + nums[r];
                    if (threeSum < 0)
                    {
                        l++;
                    }
                    else if (threeSum > 0)
                    {
                        r--;
                    }
                    else
                    {
                        var data = new List<int>() { nums[i], nums[l], nums[r] };
                        result.Add(data);
                        l++;
                        while (nums[l] == nums[l - 1] && l < r)
                        {
                            l++;
                        }
                    }
                }
            }
            return result;
        }
        public int[] CountBits(int n) //DP
        {
            var res = new int[n + 1];
            Array.Fill(res, 0);
            int offset = 1;
            for (int i = 1; i < n + 1; i++)
            {
                if (offset * 2 == i)
                    offset = i;
                res[i] = 1 + res[i - offset];
            }
            return res;
        }

        public int MissingNumber(int[] nums)
        {
            int res = nums.Length;
            for (int i = 0; i < nums.Length; i++)
            {
                res += (i - nums[i]);
            }
            return res;
        }
        public uint reverseBits(uint n)
        {
            uint ans = 0;
            for (int i = 0; i < 32; i++)
            {
                ans <<= 1;
                ans = ans | (n & 1);
                n >>= 1;
            }
            return ans;
        }
        public int ClimbStairs(int n)
        {
            int one = 1, two = 1;

            for (int i = 0; i < n - 1; i++)
            {
                int temp = one;
                one = one + two;
                two = temp;
            }
            return one;
        }
        public static int CoinChange(int[] coins, int amount)
        {
            int[] dp = new int[amount + 1];
            Array.Fill(dp, amount + 1);
            dp[0] = 0;
            for (int a = 1; a <= amount; a++)
            {
                foreach (var c in coins)
                {
                    if (a - c >= 0)
                    {
                        dp[a] = Math.Min(dp[a], 1 + dp[a - c]);
                    }
                }
            }
            return dp[amount] == amount + 1 ? -1 : dp[amount];
        }

        public bool IsAnagram(string s, string t)
        {
            if (s.Length != t.Length) return false;
            var countS = new Dictionary<string, int> { };
            var countT = new Dictionary<string, int> { };
            for (int i = 0; i < s.Length; i++)
            {
                countS[s[i].ToString()] = 1 + countS.GetValueOrDefault(s[i].ToString());
                countT[t[i].ToString()] = 1 + countT.GetValueOrDefault(t[i].ToString());
            }
            foreach (var c in countS.Keys)
            {
                countS.TryGetValue(c.ToString(), out int one);
                countT.TryGetValue(c.ToString(), out int two);
                if (one != two)
                    return false;
            }
            return true;
        }
        public int GetSum(int a, int b) //without +/-
        {
            while (b != 0)
            {
                int temp = (a & b) << 1;
                a = a ^ b;
                b = temp;
            }
            return a;
        }
        public int Trap(int[] height)
        {
            if (height.Length <= 0) return 0;
            int res = 0;
            int l = 0, r = height.Length - 1;
            int maxL = height[l], maxR = height[r];
            while (l < r)
            {
                if (maxL < maxR)
                {
                    l++;
                    maxL = Math.Max(maxL, height[l]);
                    res += maxL - height[l];
                }
                else
                {
                    r--;
                    maxR = Math.Max(maxR, height[r]);
                    res += maxR - height[r];
                }
            }
            return res;
        }

        public int LargestRectangleArea(int[] heights)
        {
            int n = heights.Length, max = 0;
            var stack = new Stack<int>();
            for (int i = 0; i <= n; i++)
            {
                var height = i < n ? heights[i] : 0;
                while (stack.Count != 0 && heights[stack.Peek()] > height)
                {
                    var currHeight = heights[stack.Pop()];
                    var prevIndex = stack.Count == 0 ? -1 : stack.Peek();
                    max = Math.Max(max, currHeight * (i - 1 - prevIndex));
                }
                stack.Push(i);
            }

            return max;

            //if (heights == null || heights.Length == 0)
            //    return 0;

            //int result = 0;
            //Stack<int> s = new Stack<int>();

            //for (int i = 0; i <= heights.Length; i++)
            //{
            //    int cur = i == heights.Length ? 0 : heights[i];

            //    while (s.Count > 0 && heights[s.Peek()] > cur)
            //    {
            //        int h = heights[s.Pop()],
            //            w = i - 1 - (s.Count == 0 ? -1 : s.Peek());

            //        result = Math.Max(result, h * w);
            //    }

            //    s.Push(i);
            //}

            //return result;
        }
        public static int LengthOfLIS(int[] nums)
        {
            int[] dp = new int[nums.Length];
            Array.Fill(dp, 1);

            for (int i = nums.Length - 1; i >= 0; i--)
            {
                for (int j = i + 1; j < nums.Length; j++)
                {
                    if (nums[i] < nums[j])
                    {
                        dp[i] = Math.Max(dp[i], 1 + dp[j]);
                    }
                }
            }
            return dp.Max();
        }

        public static int LongestCommonSubsequence(string text1, string text2)
        {
            int[,] dp = new int[text1.Length + 1, text2.Length + 1];
            for (int i = 0; i < text1.Length + 1; i++)
            {
                for (int j = 0; j < text2.Length + 1; j++)
                {
                    dp[i, j] = 0;
                }
            }
            for (int i = text1.Length - 1; i >= 0; i--)
            {
                for (int j = text2.Length - 1; j >= 0; j--)
                {
                    if (text1[i] == text2[j])
                        dp[i, j] = 1 + dp[i + 1, j + 1];
                    else
                        dp[i, j] = Math.Max(dp[i, j + 1], dp[i + 1, j]);
                }
            }
            return dp[0, 0];
        }
        public static bool WordBreak(string s, IList<string> wordDict)
        {
            bool[] dp = new bool[s.Length + 1];
            Array.Fill(dp, false);
            dp[s.Length] = true;

            for (int i = s.Length - 1; i >= 0; i--)
            {
                foreach (var w in wordDict)
                {
                    if (i + w.Length <= s.Length && s.Substring(i, w.Length) == w)
                        dp[i] = dp[i + w.Length];

                    if (dp[i] == true)
                        break;
                }
            }
            return dp[0];

            //var dp = new List<bool> {

            //}  ;
            //dp[s.Length] = true;
            //foreach (var i in Enumerable.Range(0, Convert.ToInt32(Math.Ceiling(Convert.ToDouble(-1 - (s.Length - 1)) / -1))).Select(_x_1 => s.Count - 1 + _x_1 * -1))
            //{
            //    foreach (var w in wordDict)
            //    {
            //        if (i + w.Count <= s.Count && s.Substring(i,w.Length) == w)
            //        {
            //            dp[i] = dp[i + w.Count];
            //        }
            //        if (dp[i])
            //        {
            //            break;
            //        }
            //    }
            //}
            //return dp[0];
        }

        IList<IList<int>> result = new List<IList<int>>();
        public void backtrack(int index, List<int> path, int total, int[] candidates, int target)
        {
            if (total == target)
            {
                result.Add(path.ToList());
                return;
            }

            if (total > target || index >= candidates.Length) return;

            path.Add(candidates[index]);
            backtrack(index,
                      path,
                      total + candidates[index],
                      candidates,
                      target);

            path.Remove(path.Last());

            backtrack(index + 1,
                      path,
                      total,
                      candidates,
                      target);

        }
        public IList<IList<int>> CombinationSum(int[] candidates, int target)
        {
            backtrack(0, new List<int>(), 0, candidates, target);
            return result;
        }

        public class Node
        {
            public int val;
            public IList<Node> neighbors;

            public Node()
            {
                val = 0;
                neighbors = new List<Node>();
            }

            public Node(int _val)
            {
                val = _val;
                neighbors = new List<Node>();
            }

            public Node(int _val, List<Node> _neighbors)
            {
                val = _val;
                neighbors = _neighbors;
            }
        }

        Dictionary<Node, Node> map = new Dictionary<Node, Node>();
        public Node CloneGraph(Node node)
        {
            if (node == null) return null;
            if (!map.ContainsKey(node))
            {
                map.Add(node, new Node(node.val));
                foreach (var n in node.neighbors)
                {
                    map[node].neighbors.Add(CloneGraph(n));
                }
            }

            return map[node];
        }


        public IList<string> FindItinerary(IList<IList<string>> tickets)
        {
            var map = new Dictionary<string, List<string>>();
            foreach (var ticket in tickets)
            {
                if (!map.TryGetValue(ticket[0], out var adj))
                {
                    adj = new List<string>();
                    map.Add(ticket[0], adj);
                }
                adj.Add(ticket[1]);
            }

            foreach (var adj in map.Values)
            {
                adj.Sort(Comparer<string>.Create((a, b) => string.Compare(b, a)));
            }

            var res = new Stack<string>();
            DfsVisit(map, "JFK", res);
            return res.ToList();
        }

        private void DfsVisit(Dictionary<string, List<string>> map, string src, Stack<string> ans)
        {
            if (map.TryGetValue(src, out var adj))
            {
                while (adj.Count > 0)
                {
                    var next = adj.Last();
                    adj.RemoveAt(adj.Count - 1);
                    DfsVisit(map, next, ans);
                }
            }
            ans.Push(src);
        }
        public int FindKthLargest(int[] nums, int k)
        {

            PriorityQueue<int, int> pq = new PriorityQueue<int, int>();

            for (var i = 0; i < nums.Length; i++)
            {
                if (pq.Count < k)
                    pq.Enqueue(nums[i], nums[i]);
                else
                {
                    if (nums[i] <= pq.Peek()) continue;

                    pq.Dequeue();
                    pq.Enqueue(nums[i], nums[i]);
                }
            }
            return pq.Dequeue();
        }

        public class ListNode
        {
            public int val;
            public ListNode next;
            public ListNode(int val = 0, ListNode next = null)
            {
                this.val = val;
                this.next = next;
            }
        }
        public ListNode AddTwoNumbers(ListNode l1, ListNode l2)
        {
            int carry = 0;
            ListNode dummy = new ListNode(0);
            ListNode pre = dummy;

            while (l1 != null || l2 != null || carry == 1)
            {
                int sum = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry;
                carry = sum < 10 ? 0 : 1;
                pre.next = new ListNode(sum % 10);
                pre = pre.next;

                if (l1 != null)
                {
                    l1 = l1.next;
                }

                if (l2 != null)
                {
                    l2 = l2.next;
                }
            }

            return dummy.next;
        }


        public bool CanFinish(int numCourses, int[][] prerequisites)
        {

            IDictionary<int, List<int>> preMap = new Dictionary<int, List<int>>();
            HashSet<int> visited = new HashSet<int>();
            for (int i = 0; i < numCourses; i++)
            {
                preMap.Add(i, new List<int>());
            }

            foreach (int[] course in prerequisites)
            {
                int courseToTake = course[0];
                int courseDependOn = course[1];
                preMap[courseToTake].Add(courseDependOn);
            }

            foreach (int c in Enumerable.Range(0, numCourses))
            {
                if (!DfsGraph(preMap, visited, c))
                {
                    return false;
                }
            }
            return true;
        }
        public bool DfsGraph(IDictionary<int, List<int>> preMap, HashSet<int> visited, int crs)
        {
            if (visited.Contains(crs))
            {
                return false;
            }
            if (preMap[crs] == new List<int>())
            {
                return true;
            }
            visited.Add(crs);
            foreach (var pre in preMap[crs])
            {
                if (!DfsGraph(preMap, visited, pre))
                {
                    return false;
                }
            }
            visited.Remove(crs);
            preMap[crs] = new List<int>();
            return true;
        }

        List<int> output = null;
        public int[] FindOrder(int numCourses, int[][] prerequisites)
        {
            IDictionary<int, List<int>> preMap = new Dictionary<int, List<int>>();
            HashSet<int> visited = new HashSet<int>();
            HashSet<int> cycle = new HashSet<int>();
            output = new List<int>();
            for (int i = 0; i < numCourses; i++)
            {
                preMap.Add(i, new List<int>());
            }

            foreach (int[] course in prerequisites)
            {
                int courseToTake = course[0];
                int courseDependOn = course[1];
                preMap[courseToTake].Add(courseDependOn);
            }

            foreach (int c in Enumerable.Range(0, numCourses))
            {
                if (DfsGraphTopologicalSort(preMap, visited, cycle, c) == false)
                {
                    return Array.Empty<int>();
                }
            }
            return output.ToArray();
        }

        public bool DfsGraphTopologicalSort(IDictionary<int, List<int>> preMap, HashSet<int> visited, HashSet<int> cycle, int crs)
        {
            if (cycle.Contains(crs)) return false;
            if (visited.Contains(crs)) return true;

            if (preMap[crs] == new List<int>())
            {
                return true;
            }
            cycle.Add(crs);
            foreach (var pre in preMap[crs])
            {
                if (DfsGraphTopologicalSort(preMap, visited, cycle, pre) == false)
                {
                    return false;
                }
            }
            cycle.Remove(crs);
            visited.Add(crs);
            output.Add(crs);
            return true;
        }

        public int NumIslands(char[][] grid)
        {
            int numIslands = 0;

            if (grid?.Length == 0)
                return 0;

            for (int i = 0; i < grid.Length; i++)
            {
                for (int j = 0; j < grid[i].Length; j++)
                {
                    char c = grid[i][j];
                    if (c == '1')
                    {
                        numIslands += BfsIsLand(grid, i, j);

                    }
                }

            }
            return numIslands;
        }

        public int BfsIsLand(char[][] grid, int i, int j)
        {
            if (i < 0 || i >= grid.Length || j < 0 || j >= grid[0].Length || grid[i][j] == '0')
            {
                return 0;
            }

            grid[i][j] = '0'; //to mark it as visited in iteration.
            BfsIsLand(grid, i, j + 1);
            BfsIsLand(grid, i, j - 1);
            BfsIsLand(grid, i + 1, j);
            BfsIsLand(grid, i - 1, j);
            return 1;
        }

        public static int LadderLength(string beginWord, string endWord, IList<string> wordList)
        {
            int count = 1, countInLevel = 0;
            Queue<string> queue = new Queue<string>();
            string currentWord = string.Empty;
            HashSet<string> dictionary = new HashSet<string>(wordList);
            char[] temp = null;

            queue.Enqueue(beginWord);

            while (queue.Count != 0)
            {
                countInLevel = queue.Count();

                while (countInLevel-- > 0)
                {
                    currentWord = queue.Dequeue();

                    if (currentWord == endWord)
                        return count;

                    temp = currentWord.ToCharArray();
                    for (int i = 0; i <= temp.Length - 1; i++)
                    {
                        temp = currentWord.ToCharArray();

                        for (int j = 0; j < 26; j++)
                        {
                            temp[i] = (char)('a' + j);

                            if (dictionary.Contains(new string(temp)))
                            {
                                queue.Enqueue(new string(temp));
                                dictionary.Remove(new string(temp));
                            }
                        }
                    }
                }

                count++;
            }

            return 0;
        }

        public int NetworkDelayTime(int[][] times, int n, int k)
        {

            List<(int node, int weight)>[] adjList = new List<(int node, int weight)>[n + 1];

            for (var i = 0; i <= n; i++)
            {
                adjList[i] = new List<(int node, int wht)>();
            }

            foreach (var time in times)
            {
                adjList[time[0]].Add((time[1], time[2]));
            }

            int[] visited = new int[n + 1];
            Array.Fill(visited, 0);
            PriorityQueue<int, int> queue = new();

            queue.Enqueue(k, 0);
            int res = 0;
            while (queue.TryDequeue(out int node, out int weight))
            {
                if (visited[node] == 1) continue;
                visited[node] = 1;
                res = Math.Max(res, weight);
                foreach (var adj in adjList[node])
                {
                    var totalWT = weight + adj.weight;
                    queue.Enqueue(adj.node, totalWT);
                }
            }

            int? visitedCount = visited.Where(e => e == 1)?.Count();
            return visitedCount == n ? res : -1;
        }

        private IDictionary<char, HashSet<char>> adj;
        private HashSet<char> visit;
        private HashSet<char> cycles;
        private IList<char> alienOrderOutput;

        public string AlienOrder(string[] words)
        {

            adj = new Dictionary<char, HashSet<char>>();
            visit = new HashSet<char>();
            cycles = new HashSet<char>();
            alienOrderOutput = new List<char>();

            if (!BuildAdjacencyList(words))
            {
                return "";
            }

            foreach (var c in adj.Keys)
            {
                if (!DfsAlienOrder(c))
                {
                    return "";
                }
            }

            return string.Join("", alienOrderOutput.Reverse());
        }

        private bool DfsAlienOrder(char c)
        {
            if (cycles.Contains(c))
            {
                return false;
            }
            if (visit.Contains(c))
            {
                return true;
            }

            cycles.Add(c);

            foreach (var other in adj[c])
            {
                if (!DfsAlienOrder(other))
                {
                    return false;
                }
            }

            cycles.Remove(c);
            visit.Add(c);
            alienOrderOutput.Add(c);

            return true;
        }

        private bool BuildAdjacencyList(string[] words)
        {
            foreach (var c in string.Join("", words))
            {
                if (!adj.ContainsKey(c))
                {
                    adj.Add(c, new HashSet<char>());
                }
            }

            for (var i = 1; i < words.Length; i++)
            {
                var first = words[i - 1];
                var second = words[i];
                var f = 0;
                var s = 0;

                while (f < first.Length && s < second.Length)
                {
                    if (first[f] != second[s])
                    {
                        adj[first[f]].Add(second[s]);
                        break;
                    }
                    f++;
                    s++;
                }

                if (f < first.Length && s == second.Length)
                {
                    return false;
                }
            }

            return true;
        }

        private int noOfConnectedComponents = 0;
        private int[] rank;

        public int CountComponents(int n, int[][] edges)
        {
            if (n == 0)
                return noOfConnectedComponents;

            noOfConnectedComponents = n;
            rank = new int[n];

            for (int i = 0; i < n; i++)
                rank[i] = i;

            foreach (int[] edge in edges)
                Union(edge[0], edge[1]);

            return noOfConnectedComponents;
        }

        private void Union(int x, int y)
        {
            int p1 = Find(x), p2 = Find(y);

            if (p1 != p2)
            {
                rank[p1] = p2;
                noOfConnectedComponents--;
            }
        }

        private int Find(int n)
        {
            if (rank[n] != n)
                rank[n] = Find(rank[n]);

            return rank[n];
        }

        PriorityQueue<(int, int), int> pq;
        private HashSet<(int, int)> visited;
        public int SwimInWater(int[][] grid)
        {
            int n = grid.Length;
            pq = new PriorityQueue<(int, int), int>();
            visited = new HashSet<(int, int)>();
            pq.Enqueue((0, 0), grid[0][0]);
            while (pq.TryDequeue(out var nei, out var priority))
            {
                int r = nei.Item1;
                int c = nei.Item2;
                visited.Add((r, c));

                if (r == n - 1 && c == n - 1) return priority;
                EnQueue(r + 1, c, grid, priority);
                EnQueue(r - 1, c, grid, priority);
                EnQueue(r, c + 1, grid, priority);
                EnQueue(r, c - 1, grid, priority);
            }
            return -1;
        }

        private void EnQueue(int r, int c, int[][] grid, int preCost)
        {
            if (r >= grid.Length || r < 0) return;
            if (c >= grid.Length || c < 0) return;
            if (visited.Contains((r, c))) return;

            pq.Enqueue((r, c), Math.Max(preCost, grid[r][c]));
        }

        public static int MaxScore(int[] cardPoints, int k)
        {
            if (k > cardPoints.Length)
            {
                return cardPoints.Sum();
            }
            int res = 0, total = 0;
            int l = 0, r = cardPoints.Length - k;
            for (int i = r; i < cardPoints.Length; i++)
                total += cardPoints[i];
            res = total;
            while (r < cardPoints.Length)
            {
                total += cardPoints[l] - cardPoints[r];
                res = Math.Max(res, total);
                l++;
                r++;
            }
            return res;
        }

        public bool CanAttendMeetings(int[][] intervals)
        {
            Array.Sort(intervals, (a, b) => a[0] < b[0] ? -1 : 1);
            for (int i = 1; i < intervals.Length; i++)
            {
                if (intervals[i][0] < intervals[i - 1][1])
                {
                    return false;
                }
            }
            return true;
        }

        public int MinMeetingRooms(int[][] intervals)
        {
            if (intervals == null || intervals.Length == 0)
                return 0;

            int result = 0;
            int[] start = new int[intervals.Length], end = new int[intervals.Length];

            for (int i = 0; i < intervals.Length; i++)
            {
                start[i] = intervals[i][0];
                end[i] = intervals[i][1];
            }

            Array.Sort(start);
            Array.Sort(end);

            for (int s = 0, e = 0; s < start.Length; s++)
                if (start[s] < end[e])
                    result++;
                else
                    e++;

            return result;
        }

        public IList<IList<int>> PacificAtlantic(int[][] heights)
        {
            List<IList<int>> res = new();
            int m = heights.Length, n = heights[0].Length;
            bool[,] isPacific = new bool[m, n];
            bool[,] isAtlantic = new bool[m, n];
            for (int row = 0; row < m; row++)
            {
                DFSPacificAtlantic(row, 0, heights, isPacific, heights[row][0]);
                DFSPacificAtlantic(row, n - 1, heights, isAtlantic, heights[row][n - 1]);
            }

            for (int col = 0; col < n; col++)
            {
                DFSPacificAtlantic(0, col, heights, isPacific, heights[0][col]);
                DFSPacificAtlantic(m - 1, col, heights, isAtlantic, heights[m - 1][col]);
            }
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (isAtlantic[i, j] && isPacific[i, j])
                    {
                        res.Add(new List<int> { i, j });
                    }
                }
            }

            return res;
        }

        private void DFSPacificAtlantic(int row, int col, int[][] heights, bool[,] visits, int prev)
        {
            int m = heights.Length, n = heights[0].Length;
            if (row < 0 || row >= m || col < 0 || col >= n || visits[row, col] || heights[row][col] < prev)
                return;
            visits[row, col] = true;
            DFSPacificAtlantic(row, col + 1, heights, visits, heights[row][col]);
            DFSPacificAtlantic(row, col - 1, heights, visits, heights[row][col]);
            DFSPacificAtlantic(row + 1, col, heights, visits, heights[row][col]);
            DFSPacificAtlantic(row - 1, col, heights, visits, heights[row][col]);
        }


        int perimeter = 0;
        public int IslandPerimeter(int[][] grid)
        {
            if (grid?.Length == 0)
                return 0;

            bool[,] visits = new bool[grid.Length, grid[0].Length];

            for (int i = 0; i < grid.Length; i++)
            {
                for (int j = 0; j < grid[0].Length; j++)
                {
                    if (grid[i][j] == 1)
                    {
                        return DfsIslandPerimeter(grid, visits, i, j);
                    }
                }

            }

            return perimeter;
        }

        public int DfsIslandPerimeter(int[][] grid, bool[,] visits, int i, int j)
        {
            if (i < 0 || i >= grid.Length || j < 0 || j >= grid[0].Length || grid[i][j] == 0)
            {
                return 1;
            }
            if (visits[i, j])
            {
                return 0;
            }

            visits[i, j] = true; //to mark it as visited in iteration.
            perimeter = DfsIslandPerimeter(grid, visits, i, j + 1);
            perimeter += DfsIslandPerimeter(grid, visits, i + 1, j);
            perimeter += DfsIslandPerimeter(grid, visits, i, j - 1);
            perimeter += DfsIslandPerimeter(grid, visits, i - 1, j);
            return perimeter;
        }

        public int MinReorder(int n, int[][] connections)
        {
            if (connections == null || connections?.Length < 2) return 0;

            Dictionary<int, HashSet<int>> paths = new Dictionary<int, HashSet<int>>();
            List<int>[] graph = new List<int>[n];
            foreach (var connection in connections)
            {
                if (!paths.ContainsKey(connection[0]))
                    paths.Add(connection[0], new HashSet<int>());

                paths[connection[0]].Add(connection[1]);

                if (graph[connection[0]] == null)
                    graph[connection[0]] = new List<int>();
                graph[connection[0]].Add(connection[1]);

                if (graph[connection[1]] == null)
                    graph[connection[1]] = new List<int>();
                graph[connection[1]].Add(connection[0]);
            }
            int cnt = 0;
            HashSet<int> visited = new HashSet<int>();
            DFSMinReorder(graph, 0, paths, visited, ref cnt);
            return cnt;
        }

        private void DFSMinReorder(List<int>[] graph, int u, Dictionary<int, HashSet<int>> paths, HashSet<int> visited, ref int cnt)
        {
            visited.Add(u);

            if (graph[u] != null)
            {
                foreach (var v in graph[u])
                {
                    if (!visited.Contains(v))
                    {
                        if (paths.ContainsKey(u) && paths[u].Contains(v))
                            cnt++;

                        DFSMinReorder(graph, v, paths, visited, ref cnt);
                    }
                }
            }
        }

        public int MaxAreaOfIsland(int[][] grid)
        {
            int r = grid.Length, c = grid[0].Length, area = 0;

            bool[,] visits = new bool[r, c];
            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    area = Math.Max(area, DFSMaxAreaOfIsland(i, j, grid, visits));
                }
            }

            return area;
        }

        private int DFSMaxAreaOfIsland(int row, int col, int[][] grid, bool[,] visits)
        {
            int m = grid.Length, n = grid[0].Length;
            if (row < 0 || row >= m || col < 0 || col >= n || visits[row, col] || grid[row][col] == 0)
                return 0;
            visits[row, col] = true;
            return (1 + DFSMaxAreaOfIsland(row, col + 1, grid, visits) +
             DFSMaxAreaOfIsland(row, col - 1, grid, visits) +
             DFSMaxAreaOfIsland(row + 1, col, grid, visits) +
             DFSMaxAreaOfIsland(row - 1, col, grid, visits));
        }

        public int SnakesAndLadders(int[][] board)
        {
            int n = board.Length;
            bool[,] visits = new bool[n, n];
            Queue<int> q = new Queue<int>();
            q.Enqueue(1);
            int res = 0;

            while (q.Count > 0)
            {
                int cnt = q.Count;

                for (int i = 0; i < cnt; i++)
                {
                    int cur = q.Dequeue();
                    if (cur == n * n)
                        return res;

                    for (int j = 1; j <= 6; j++)
                    {
                        int newVal = cur + j;

                        if (newVal > n * n)
                            break;
                        var pos = GetPositionForInt(newVal, n);
                        if (visits[pos.r, pos.c] == true)
                            continue;
                        if (board[pos.r][pos.c] == -1)
                            q.Enqueue(newVal);
                        else
                            q.Enqueue(board[pos.r][pos.c]);

                        visits[pos.x, pos.y] = true;
                    }
                }
                res++;
            }

            return -1;
        }
        private (int r, int c) GetPositionForInt(int x, int n)
        {
            int r = n - (x - 1) / n - 1;
            int c = (x - 1) % n;
            if (r % 2 == n % 2)
                return (r, n - c - 1);
            return (r, c);
        }


        public int OpenLock(string[] deadends, string target)
        {
            const string start = "0000";

            var deadEnds = deadends.ToHashSet();
            var visited = new HashSet<string>();
            if (deadEnds.Contains(start) || deadends.Contains(target)) return -1;

            Queue<string> q = new Queue<string>(new[] { start });
            visited.Add(start);
            int res = 0;
            while (q.Count > 0)
            {
                int queueCnt = q.Count;
                for (int i = 0; i < queueCnt; i++)
                {
                    var curr = q.Dequeue();
                    if (curr == target) return res;
                    foreach (var nei in GetNeighbors(curr))
                    {
                        if (!visited.Contains(nei) && !deadends.Contains(nei))
                        {
                            q.Enqueue(nei);
                            visited.Add(nei);
                        }
                    }
                }
                res++;
            }
            return -1;
        }

        private List<string> GetNeighbors(string s)
        {
            var result = new List<string>();
            for (int i = 0; i < s.Length; i++)
            {
                var charAr1 = s.ToCharArray();
                charAr1[i] = charAr1[i] == '9' ? '0' : (char)(charAr1[i] + 1);
                result.Add(new string(charAr1));
                var charAr2 = s.ToCharArray();
                charAr2[i] = charAr2[i] == '0' ? '9' : (char)(charAr2[i] - 1);
                result.Add(new string(charAr2));
            }
            return result;
        }

        public int OrangesRotting(int[][] grid)
        {
            if (grid == null || grid[0].Length == 0)
                return 0;

            int r = grid.Length, c = grid[0].Length, fresh = 0, time = 0;

            Queue<(int, int)> q = new Queue<(int, int)>();

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    if (grid[i][j] == 1)
                        fresh++;
                    else if (grid[i][j] == 2)
                    {
                        q.Enqueue((i, j));
                    };
                }
            }

            if (fresh == 0)
                return 0;

            int[,] dir = new int[,] { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
            while (q.Any())
            {
                time++;
                int size = q.Count;
                for (int i = 0; i < size; i++)
                {
                    var curr = q.Dequeue();
                    for (int j = 0; j < 4; j++)
                    {
                        int row = curr.Item1 + dir[j, 0];
                        int col = curr.Item2 + dir[j, 1];

                        if (row >= 0 && row < r && col >= 0 && col < c && grid[row][col] == 1)
                        {
                            grid[row][col] = 2;
                            q.Enqueue((row, col));
                            fresh--;
                        }
                    }
                }
            }

            return fresh == 0 ? time - 1 : -1;
        }

        public bool CarPooling(int[][] trips, int capacity)
        {
            var dictionary = new SortedDictionary<int, int>();

            foreach (var trip in trips)
            {
                if (!dictionary.ContainsKey(trip[1])) dictionary[trip[1]] = 0;
                if (!dictionary.ContainsKey(trip[2])) dictionary[trip[2]] = 0;
                dictionary[trip[1]] += trip[0]; dictionary[trip[2]] -= trip[0];
            }
            int currentPassennger = 0;
            foreach (var keyval in dictionary)
            {
                currentPassennger += keyval.Value;
                if (currentPassennger > capacity) return false;
            }
            return true;
        }

        public int CountSubIslands(int[][] grid1, int[][] grid2)
        {
            int r = grid1.Length, c = grid1[0].Length, subIslands = 0;
            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < grid2[r].Length; j++)
                {
                    if (grid2[i][j] == 1 && DfsSubIsLand(grid1, grid2, r, c))
                    {
                        subIslands++;
                    }
                }
            }
            return subIslands;
        }

        private bool DfsSubIsLand(int[][] grid1, int[][] grid2, int r, int c)
        {
            if (r < 0 || c < 0 || r >= grid2.Length || c >= grid2[r].Length || grid2[r][c] == 0) { return true; }

            grid2[r][c] = 0;
            bool stillASubIsland = true;
            stillASubIsland &= DfsSubIsLand(grid1, grid2, r - 1, c);
            stillASubIsland &= DfsSubIsLand(grid1, grid2, r, c + 1);
            stillASubIsland &= DfsSubIsLand(grid1, grid2, r + 1, c);
            stillASubIsland &= DfsSubIsLand(grid1, grid2, r, c - 1);

            return stillASubIsland && grid1[r][c] == 1;
        }

        public IList<IList<string>> SolveNQueens(int n)
        {
            var result = new List<IList<string>>();
            IList<StringBuilder> board = new List<StringBuilder>();
            for (int i = 0; i < n; i++)
            {
                board.Add(new StringBuilder(n));
                board[i].Append('.', n);

            }
            backtrackingNQueens(n, 0, board, result, new HashSet<(int i, int j)>(), new HashSet<(int i, int j)>(), new HashSet<(int i, int j)>());
            return result;
        }

        private void backtrackingNQueens(int n, int row, IList<StringBuilder> board, List<IList<string>> result, HashSet<(int i, int j)> col, HashSet<(int i, int j)> positiveDiag, HashSet<(int i, int j)> negativeDiag)
        {
            if (n == 0)
            {
                result.Add(board.Select(s => s.ToString()).ToList());
                return;
            }
            if (row == board.Count) return;

            for (int c = 0; c < board.Count; c++)
            {
                (int i, int j) column = (0, c);
                int m = Math.Min(row, c);
                (int i, int j) diag1 = (row - m, c - m);
                m = Math.Min(row, board.Count - 1 - c);
                (int i, int j) diag2 = (row - m, c + m);

                if (col.Contains(column) || positiveDiag.Contains(diag1) ||
                   negativeDiag.Contains(diag2)) continue;

                col.Add(column);
                positiveDiag.Add(diag1);
                negativeDiag.Add(diag2);


                board[row][c] = 'Q';
                backtrackingNQueens(n - 1, row + 1, board, result, col, positiveDiag, negativeDiag);

                board[row][c] = '.';
                col.Remove(column);
                positiveDiag.Remove(diag1);
                negativeDiag.Remove(diag2);
            }
        }

        public int UniquePaths(int m, int n)
        {
            var row = new int[n];
            Array.Fill(row, 1);

            foreach (var i in Enumerable.Range(0, m - 1))
            {
                var newRow = new int[n];
                Array.Fill(newRow, 1);
                for (int j = n - 2; j > 0; j--)
                {
                    newRow[j] = newRow[j + 1] + row[j];
                }
                row = newRow;
            }
            return row[0];
        }

        public bool CanJump(int[] nums)
        {
            int goal = nums.Length - 1;
            for (int i = nums.Length - 1; i >= 0; i--)
            {
                if (nums[i] + i >= goal)
                {
                    goal = i;
                }
            }

            return goal == 0;
        }

        public int LongestConsecutive(int[] nums)
        {
            if (nums.Length < 2) return nums.Length;

            int longest = 0;
            var set = new HashSet<int>(nums);

            foreach (var item in set)
            {
                if (!set.Contains(item - 1))
                {
                    var length = 0;
                    while (set.Contains(item + length))
                    {
                        length++;
                        longest = Math.Max(longest, length);

                    }
                }
            }
            return longest;
        }
        public bool ValidTree(int n, int[][] edges)
        {
            if (n == 0) return true;

            var adj = new HashSet<int>[n];

            for (int i = 0; i < n; i++)
            {
                adj[i] = new HashSet<int>();
            }
            foreach (var edge in edges)
            {
                var e1 = edge[0];
                var e2 = edge[1];
                adj[e1].Add(e2);
                adj[e2].Add(e1);
            }
            var visited = new bool[n];

            var res = DfsValidTree(adj, 0, visited);

            if (visited.Any(c => !c)) return false;
            return res;
        }

        private bool DfsValidTree(HashSet<int>[] adj, int current, bool[] visited)
        {
            if (visited[current]) return false;
            visited[current] = true;

            var nextLevel = adj[current];
            foreach (var level in nextLevel)
            {
                adj[level].Remove(current);
                if (!DfsValidTree(adj, level, visited))
                {
                    return false;
                }
            }
            return true;
        }

        public int[][] Insert(int[][] intervals, int[] newInterval)
        {
            var result = new List<int[]>();
            int i = 0;
            while (i < intervals.Length && intervals[i][1] < newInterval[0])
                result.Add(intervals[i++]);


            while (i < intervals.Length && intervals[i][0] <= newInterval[1])
            {
                newInterval[0] = Math.Min(newInterval[0], intervals[i][0]);
                newInterval[1] = Math.Max(newInterval[1], intervals[i][1]);
                i++;
            }
            result.Add(newInterval);
            while (i < intervals.Length)
                result.Add(intervals[i++]);

            return result.ToArray();
        }
        public int[][] Merge(int[][] intervals)
        {
            if (intervals == null || intervals.Length == 0 || intervals.Length == 1)
                return intervals;

            List<int[]> res = new List<int[]>();

            intervals = intervals.OrderBy(x => x[0]).ToArray();
            int s = intervals[0][0],
            e = intervals[0][1];
            for (int i = 1; i < intervals.Length; i++)
            {
                if (intervals[i][0] > e)
                {
                    res.Add(new int[] { s, e });
                    s = intervals[i][0];
                    e = intervals[i][1];
                }
                else
                    e = Math.Max(e, intervals[i][1]);
            }

            res.Add(new int[] { s, e });

            return res.ToArray();
        }
        public int EraseOverlapIntervals(int[][] intervals)
        {
            if (intervals.Length == 0)
                return 0;
            Array.Sort(intervals, (x, y) => x[1].CompareTo(y[1]));
            int res = 0;
            int end = intervals[0][1];
            for (int i = 1; i < intervals.Length; i++)
            {
                if (intervals[i][0] > end)
                {
                    end = intervals[i][0];
                }
                else res++;
            }
            return res;
        }

        public bool HasCycle(ListNode head)
        {
            if (head == null) return false;

            ListNode slow = head, fast = head.next;

            while (slow != fast)
            {
                if (fast == null || fast.next == null)
                {
                    return false;
                }
                slow = slow.next;
                fast = fast.next.next;
            }
            return true;
        }
        public ListNode MergeTwoLists(ListNode list1, ListNode list2)
        {
            /* if (list1 == null) return list2;
             if (list2 == null) return list1;
             if (list1.val < list2.val)
             {
                 list1.next = MergeTwoLists(list1.next, list2);
                 return list1;
             }
             list2.next = MergeTwoLists(list1, list2.next);
             return list2;*/
            ListNode dummy = new ListNode();
            ListNode tail = dummy;

            while (list1 != null && list2 != null)
            {
                if (list1.val < list2.val)
                {
                    tail.next = list1;
                    list1 = list1.next;
                }
                else
                {
                    tail.next = list2;
                    list2 = list2.next;
                }
                tail = tail.next;
            }
            if (list1 != null)
            {
                tail.next = list1;
            }
            else if (list2 != null)
            {
                tail.next = list2;
            }
            return dummy.next;
        }
        public ListNode MergeKLists(ListNode[] lists)
        {
            if (lists == null || lists.Length == 0) return null;

            return MergeListsByBinSearch(lists, 0, lists.Length - 1);
        }

        private ListNode MergeListsByBinSearch(ListNode[] lists, int i, int j)
        {
            if (j == i)
            {
                return lists[i];
            }
            else
            {
                int mid = i + (j - i) / 2;

                var left = MergeListsByBinSearch(lists, i, mid);
                var right = MergeListsByBinSearch(lists, mid + 1, j);
                return MergeTwoLists(left, right);
            }
        }

        public IList<int> CountSmaller(int[] nums)
        {
            var sortedNums = new List<int>();
            var result = new int[nums.Length];

            for (int i = nums.Length - 1; i >= 0; i--)
            {
                int left = 0;
                int right = sortedNums.Count;

                while (left < right)
                {
                    var mid = left + (right - left) / 2;
                    if (sortedNums[mid] >= nums[i]) right = mid;
                    else left = mid + 1;
                }

                result[i] = left;
                sortedNums.Insert(left, nums[i]);
            }

            return result;
        }

        public int peakIndexInMountainArray(int[] arr)
        {
            int lo = 0, hi = arr.Length - 1;
            while (lo < hi)
            {
                int mi = lo + (hi - lo) / 2;
                if (arr[mi] < arr[mi + 1])
                    lo = mi + 1;
                else
                    hi = mi;
            }
            return lo;
        }
        public ListNode removeNthFromEnd(ListNode head, int n)
        {
            ListNode dummy = new ListNode(0)
            {
                next = head
            };
            int length = 0;
            ListNode first = head;
            while (first != null)
            {
                length++;
                first = first.next;
            }
            length -= n;
            first = dummy;
            while (length > 0)
            {
                length--;
                first = first.next;
            }
            first.next = first.next.next;

            char.IsLetterOrDigit
            return dummy.next;
        }
        public void ReorderList(ListNode head)
        {
            ListNode slow = head, fast = head.next;
            //middle
            while (fast?.next != null)
            {
                slow = slow.next;
                fast = fast.next.next;
            }

            //reverse
            var second = slow.next;
            ListNode pre = null;
            slow.next = null;
            while (second != null)
            {
                ListNode temp = second.next;
                second.next = pre;
                pre = second;
                second = temp;
            }

            //merge
            var first = head;
            second = pre;
            while (second != null)
            {
                var temp1 = first.next;
                var temp2 = second.next;
                first.next = second;
                second.next = temp1;
                first = temp1;
                second = temp2;
            }
        }

        public IList<int> SpiralOrder(int[][] matrix)
        {
            List<int> result = new List<int>();
            int top = 0;
            int left = 0;
            int right = matrix[0].Length - 1;
            int bottom = matrix.Length - 1;

            while (true)
            {
                //Left to Right
                for (int i = left; i <= right; i++)
                {
                    result.Add(matrix[top][i]);
                }
                top++;
                if (left > right || top > bottom) break;

                //Top to Bottom
                for (int i = top; i <= bottom; i++)
                {
                    result.Add(matrix[i][right]);
                }
                right--;
                if (left > right || top > bottom) break;

                //Right to Left
                for (int i = right; i >= left; i--)
                {
                    result.Add(matrix[bottom][i]);
                }
                bottom--;
                if (left > right || top > bottom) break;

                //Bottom to Top
                for (int i = bottom; i >= top; i--)
                {
                    result.Add(matrix[i][left]);
                }
                left++;
                if (left > right || top > bottom) break;
            }//Repeat
            return result;
        }

        public int CountSubstrings(string s)
        {
            int res = 0;
            for (int i = 0; i < s.Length; i++)
            {
                res += CountPali(s, i, i);
                res += CountPali(s, i, i + 1);
            }
            return res;
        }

        private int CountPali(string s, int i, int v)
        {
            int res = 0;
            while (i >= 0 && v < s.Length && s[i] == s[v])
            {
                res++;
                i--; v++;
            }
            return res;
        }


        int maxPathSum = Int32.MinValue;

        public int MaxPathSum(TreeNode root)
        {
            DfsMaxPathSum(root);
            return maxPathSum;
        }

        private int DfsMaxPathSum(TreeNode root)
        {
            if (root == null)
                return 0;

            int leftMax = DfsMaxPathSum(root.left),
                rightMax = DfsMaxPathSum(root.right),
                currentMax = 0;

            currentMax = Math.Max(currentMax, Math.Max(leftMax + root.val, rightMax + root.val));
            maxPathSum = Math.Max(maxPathSum, leftMax + root.val + rightMax);

            return currentMax;
        }

        public int[] TopKFrequent(int[] nums, int k)
        {
            int[] arr = new int[k];
            var dict = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                if (dict.ContainsKey(nums[i]))
                {
                    dict[nums[i]]++;
                }
                else
                {
                    dict.Add(nums[i], 1);
                }
            }

            var pq = new PriorityQueue<int, int>();
            foreach (var key in dict.Keys)
            {
                pq.Enqueue(key, dict[key]);
                if (pq.Count > k) pq.Dequeue();
            }
            int i2 = k;
            while (pq.Count > 0)
            {
                arr[--i2] = pq.Dequeue();
            }
            return arr;
        }

        public class Engineer
        {
            public int speed;
            public int efficiency;
            public Engineer(int speed, int efficiency)
            {
                this.speed = speed;
                this.efficiency = efficiency;
            }

        }

        public int MaxPerformance(int n, int[] speed, int[] efficiency, int k)
        {
            List<Engineer> engineers = new();
            for (int i = 0; i < n; i++)
            {
                engineers.Add(new Engineer(speed[i], efficiency[i]));
            }

            engineers = engineers.OrderByDescending(x => x.efficiency).ToList();
            var queue = new PriorityQueue<int, int>();
            long speedTotal = 0, result = 0;
            foreach (var engineer in engineers)
            {
                if (queue.Count > k - 1)
                    speedTotal -= queue.Dequeue();
                queue.Enqueue(engineer.speed, engineer.speed);
                speedTotal += engineer.speed;
                result = Math.Max(result, speedTotal * engineer.efficiency);
            }

            return (int)(result % 1000000007);
        }

        public IList<string> LetterCombinations(string digits)
        {
            var lettersMap = new Dictionary<char, string>
            {
                {'2', "abc"},
                {'3', "def"},
                {'4', "ghi"},
                {'5', "jkl"},
                {'6', "mno"},
                {'7', "pqrs"},
                {'8', "tuv"},
                {'9', "wxyz"}
            };

            var result = new List<string>();

            if (!string.IsNullOrEmpty(digits))
                Backtrack(result, digits, lettersMap, "", 0);

            return result;
        }

        void Backtrack(List<string> result, string digits, Dictionary<char, string> lettersMap, string curString, int start)
        {
            if (curString.Length == digits.Length)
            { result.Add(curString); return; }

            foreach (var c in lettersMap[digits[start]])
            {
                Backtrack(result, digits, lettersMap, curString + c, start + 1);
            }
        }

        public bool BtreeGameWinningMove(TreeNode root, int n, int x)
        {
            if (root == null) return false;

            if (root.val == x)
            {
                int leftCount = count(root.left);
                int rightCount = count(root.right);
                int parentCount = n - (leftCount + rightCount + 1);

                return parentCount > (leftCount + rightCount) || leftCount > (parentCount + rightCount) || rightCount > (leftCount + parentCount);
            }

            return BtreeGameWinningMove(root.left, n, x) || BtreeGameWinningMove(root.right, n, x);
        }

        private int count(TreeNode node)
        {
            if (node == null) return 0;
            return count(node.left) + count(node.right) + 1;
        }

        public int MaximumInvitations(int[][] grid)
        {
            int m = grid.Length; // boys
            int n = grid[0].Length; // girls

            int[] girlFixed = new int[n];

            for (int i = 0; i < n; i++)
            {
                girlFixed[i] = -1;
            }

            int invitations = 0;

            for (int i = 0; i < m; i++)
            {
                var seenGirl = new HashSet<int>();

                if (dfsInvitations(grid, i, seenGirl, girlFixed))
                {
                    invitations++;
                }
            }
            return invitations;
        }

        private bool dfsInvitations(int[][] grid, int boy, HashSet<int> seenGirl, int[] girlFixed)
        {
            int m = grid.Length; // boys
            int n = grid[0].Length; // girls

            for (int i = 0; i < n; i++)
            {
                if (grid[boy][i] == 1 && !seenGirl.Contains(i))
                {
                    seenGirl.Add(i);
                    if (girlFixed[i] == -1 || dfsInvitations(grid, girlFixed[i], seenGirl, girlFixed))
                    {
                        girlFixed[i] = boy;
                        return true;
                    }
                }
            }
            return false;
        }

        //2128
        public bool RemoveOnes(int[][] grid)
        {
            if (grid == null || grid.Length == 0)
                throw new ArgumentException("Invalid Input.");

            int m = grid.Length, n = grid[0].Length;
            for (int i = 1; i < m; i++)
            {
                for (int j = 1; j < n; j++)
                {
                    if ((grid[i][j] ^ grid[i - 1][j]) != (grid[i][j - 1] ^ grid[i - 1][j - 1]))
                        return false;
                }
            }

            return true;
        }

        //562
        public int LongestLine(int[][] mat)
        {
            if (mat == null || mat.Length == 0 || mat[0].Length == 0) return 0;
            int r = mat.Length, c = mat[0].Length, result = 0;
            var dp = new int[r + 2, c + 2, 4];
            for (int i = 1; i <= r; i++)
            {
                for (int j = 1; j <= c; j++)
                {
                    if (mat[i - 1][j - 1] == 1)
                    {
                        dp[i, j, 0] = dp[i - 1, j, 0] + 1;
                        dp[i, j, 1] = dp[i - 1, j - 1, 1] + 1;
                        dp[i, j, 2] = dp[i, j - 1, 2] + 1;
                    }
                }
            }

            for (int i = 1; i <= r; i++)
            {
                for (int j = c; j >= 1; j--)
                {
                    if (mat[i - 1][j - 1] == 1)
                    {
                        dp[i, j, 3] = dp[i - 1, j + 1, 3] + 1;
                        result = new int[] { result, dp[i, j, 0], dp[i, j, 1], dp[i, j, 2], dp[i, j, 3] }.Max();
                    }
                }
            }

            return result;
        }

        //1048
        public int LongestStrChain(string[] words)
        {
            var dic = new Dictionary<string, int>();
            foreach (var word in words.OrderBy(w => w.Length))
                for (int i = 0; i < word.Length; i++)
                {
                    string word2 = word.Remove(i, 1);
                    dic[word] = Math.Max(dic.ContainsKey(word) ? dic[word] : 1,
                        dic.ContainsKey(word2) ? dic[word2] + 1 : 1);
                }

            return dic.Values.Max();
        }

        public bool DifferByOne(string[] dict)
        {
            HashSet<string> pSet = new HashSet<string>();
            char[][] charArray = new char[dict.Length][];

            for (int i = 0; i < dict.Length; i++)
            {
                charArray[i] = dict[i].ToCharArray();
            }

            for (int i = 0; i < dict[0].Length; i++)
            {
                pSet.Clear();
                for (int j = 0; j < dict.Length; j++)
                {
                    char tmp = charArray[j][i];
                    charArray[j][i] = '*';
                    var newS = new string(charArray[j]);
                    charArray[j][i] = tmp;

                    if (pSet.Contains(newS))
                    {
                        return true;
                    }

                    pSet.Add(newS);
                }
            }

            return false;
        }

        public int shortestWay(String source, String target)
        {
            int j = 0; // Pointer in target
            int minSubseq = 0;

            while (j < target.Length)
            {
                bool matched = false; // When true, subsequences of source can match target 
                for (int i = 0; i < source.Length && j < target.Length; i++)
                {
                    if (source[i] == target[j])
                    {
                        j++;
                        matched = true;
                    }
                }
                if (!matched)
                    return -1;
                minSubseq++;
            }
            return minSubseq;
        }

        public interface Robot
        {
            // Returns true if the cell in front is open and robot moves into the cell.
            // Returns false if the cell in front is blocked and robot stays in the current cell.
            public bool Move();

            // Robot will stay in the same cell after calling turnLeft/turnRight.
            // Each turn will be 90 degrees.
            public void TurnLeft();
            public void TurnRight();

            // Clean the current cell.
            public void Clean();
        }
        private readonly (int, int)[] directions = { (-1, 0), (0, 1), (1, 0), (0, -1) };
        //private ISet<(int, int)> visited = new HashSet<(int, int)>();
        private Robot robot;

        public void CleanRoom(Robot robot)
        {
            this.robot = robot;
            Backtrack(0, 0, 0);
        }

        public void GoBack()
        {
            robot.TurnRight();
            robot.TurnRight();
            robot.Move();
            robot.TurnRight();
            robot.TurnRight();
        }

        public void Backtrack(int row, int col, int direction)
        {
            robot.Clean();
            visited.Add((row, col));
            for (int i = 0; i < 4; i++)
            {
                int newD = (direction + i) % 4;
                int newRow = row + directions[newD].Item1;
                int newCol = col + directions[newD].Item2;

                if (!visited.Contains((newRow, newCol)) && robot.Move())
                {
                    Backtrack(newRow, newCol, newD);
                    GoBack();
                }

                robot.TurnRight();
            }
        }

        public class Interval
        {
            public int start;
            public int end;

            public Interval() { }
            public Interval(int _start, int _end)
            {
                start = _start;
                end = _end;
            }
        }
        public IList<Interval> EmployeeFreeTime(IList<IList<Interval>> schedule)
        {
            List<Interval> intervals = schedule.SelectMany(sch => sch).ToList();

            intervals.Sort((a, b) => a.start.CompareTo(b.start));
            List<Interval> result = new List<Interval>();
            if (intervals.Count <= 1)
            {
                return result;
            }

            int maxEnd = intervals[0].end;
            for (int i = 1; i < intervals.Count; i++)
            {
                if (intervals[i].start > maxEnd)
                {
                    result.Add(new Interval(maxEnd, intervals[i].start));
                }

                maxEnd = Math.Max(maxEnd, intervals[i].end);
            }

            return result;
        }

        public int wordsTyping(String[] sentence, int rows, int cols)
        {
            int currLength = 0, wordNo = 0, timesFit = 0;

            int spaceRequired = 0; // The length needed for a complete sentence to be fitted.
            foreach (var word in sentence)
            {
                spaceRequired += word.Length + 1;
            }
            int rowTimesFit = (cols + 1) / spaceRequired; // The default fit times for a row.
            int defaultLength = rowTimesFit * spaceRequired;

            for (int i = 0; i < rows; i++)
            {
                currLength = defaultLength;
                timesFit += rowTimesFit;
                while (currLength + sentence[wordNo].Length <= cols)
                {
                    currLength += sentence[wordNo].Length;
                    if (currLength < cols)
                        currLength++;
                    wordNo++;
                    if (wordNo == sentence.Length)
                    {
                        wordNo = 0;
                        timesFit++;
                    }
                }
            }

            return timesFit;
        }
         
        public int MinCostClimbingStairs(int[] cost)
        {
            cost.Append(0);
            for (int i = cost.Length - 2; i >= 0; i--)
            {
                cost[i] += Math.Min(cost[i + 1], cost[i + 2]);
            }

            return Math.Min(cost[0], cost[1]);
        }

        public int Rob(int[] nums)
        {
            int rob1 = 0, rob2 = 0;
            foreach (var n in nums)
            {
                int temp = Math.Max(rob1 + n, rob2);
                rob1 = rob2;
                rob2 = temp;
            }
            return rob2;
        }

        public int RobHelper(int[] nums)
        {
            int rob1 = 0, rob2 = 0;
            foreach (var n in nums)
            {
                int temp = Math.Max(rob1 + n, rob2);
                rob1 = rob2;
                rob2 = temp;
            }
            return rob2;
        }

        public int Rob2(int[] nums)
        {
            var n = nums.Length;
            if (n == 0) return 0;
            if (n == 1) return nums[0];

            var numList = new List<int>(nums);
            numList.RemoveAt(0);
            var candidate1 = RobHelper(numList.ToArray());

            numList = new List<int>(nums);
            numList.RemoveAt(n - 1);
            var cnadidate2 = RobHelper(numList.ToArray());
            return Math.Max(candidate1, cnadidate2);
        }

        public int NumDecodings(string s)
        {
            List<int> dp = Enumerable.Repeat(1, s.Length + 1).ToList();

            for (var i = s.Length - 1; i >= 0; i--)
            {
                if (s[i] == '0')
                    dp[i] = 0;
                else
                    dp[i] = dp[i + 1];

                if (i + 1 < s.Length && (
                    s[i] == '1' ||
                    (s[i] == '2' && "0123456".Contains(s[i + 1]))
                ))
                    dp[i] += dp[i + 2];
            }

            return dp[0];
        }
    }
}

class MedianFinder
{

    private PriorityQueue<int, int> smallHeap; //small elements - maxHeap
    private PriorityQueue<int, int> largeHeap; //large elements - minHeap

    public MedianFinder()
    {
        smallHeap = new PriorityQueue<int, int>();
        largeHeap = new PriorityQueue<int, int>();
    }

    public void addNum(int num)
    {
        smallHeap.Enqueue(num, num);
        if (
            smallHeap.Count - largeHeap.Count > 1 ||
            !(largeHeap.Count <= 0) &&
            smallHeap.Peek() > largeHeap.Peek()
        )
        {
            if (smallHeap.Count > 0)
            {
                int ele = smallHeap.Dequeue();
                largeHeap.Enqueue(ele, ele);
            }
        }
        if (largeHeap.Count - smallHeap.Count > 1)
        {
            if (largeHeap.Count > 0)
            {
                int ele = largeHeap.Dequeue();
                smallHeap.Enqueue(ele, ele);
            }
        }
    }

    public double findMedian()
    {
        if (smallHeap.Count == largeHeap.Count)
        {
            return (double)(largeHeap.Peek() + smallHeap.Peek()) / 2;
        }
        else if (smallHeap.Count > largeHeap.Count)
        {
            return (double)smallHeap.Peek();
        }
        else
        {
            return (double)largeHeap.Peek();
        }
    }
}

public class MedianFinderBin
{
    List<int> Nums;
    public MedianFinderBin()
    {
        Nums = new List<int>();
    }

    public void AddNum(int num)
    {
        int index = Nums.BinarySearch(num);
        if (index < 0)
        {
            index = ~index;
        }
        Nums.Insert(index, num);
    }

    public double FindMedian()
    {
        int count = Nums.Count;
        return count % 2 == 0 ? (double)((Nums[count / 2 - 1] + Nums[count / 2]) * 0.5) : Nums[count / 2];
    }
}
}