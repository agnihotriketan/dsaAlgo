using System;
using System.Collections;
using System.Diagnostics.Metrics;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;

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
            //var median = FindMedianSortedArrays(ar1, ar2);
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
            int noCompo = CountComponents(5, edges);
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
            var A = nums1;
            var B = nums2;

            var total = nums1.Length + nums2.Length;
            var half = total / 2;
            if (B.Length < A.Length)
            {
                A = B;
                B = A;
            }
            var l = 0;
            var r = A.Length - 1;
            while (true)
            {
                var i = (l + r) / 2;
                var j = half - i - 2;
                var Aleft = i >= 0 ? A[i] : double.NegativeInfinity;
                var Aright = i + 1 < A.Length ? A[i + 1] : double.PositiveInfinity;
                var Bleft = j >= 0 ? B[j] : double.NegativeInfinity;
                var Bright = j + 1 < B.Length ? B[j + 1] : double.PositiveInfinity;

                if (Aleft <= Bright && Bleft <= Aright)
                {
                    if (total % 2 != 0)
                    {
                        return Math.Min(Aright, Bright);
                    }
                    return (Math.Max(Aleft, Bleft) + Math.Min(Aright, Bright)) / 2;
                }
                else if (Aleft > Bright)
                {
                    r = i - 1;
                }
                else
                {
                    l = i + 1;
                }
            }
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

        private static int noOfConnectedComponents = 0;
        private static int[] rank;

        public static int CountComponents(int n, int[][] edges)
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

        private static void Union(int x, int y)
        {
            int p1 = Find(x), p2 = Find(y);

            if (p1 != p2)
            {
                rank[p1] = p2;
                noOfConnectedComponents--;
            }
        }

        private static int Find(int n)
        {
            if (rank[n] != n)
                rank[n] = Find(rank[n]);

            return rank[n];
        }
    }
}