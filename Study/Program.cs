using System;
using System.Diagnostics.Metrics;
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

            ar1 = new int[] { 1, 2, 5 };
            var ans = CoinChange(ar1, 11);
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

    }
}