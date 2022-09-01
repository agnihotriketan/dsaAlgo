using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Study
{
    public class TrieNode
    {
        public TrieNode()
        {
            childrenMap = new Dictionary<char, TrieNode>();
        }
        public Dictionary<char, TrieNode> childrenMap { get; set; }
        public bool isWord { get; set; }
    }
    public class WordDictionary
    {

        private TrieNode root;
        public WordDictionary()
        {
            root = new TrieNode();
        }

        public void AddWord(string word)
        {
            var cur = root;
            foreach (var c in word)
            {
                if (!cur.childrenMap.ContainsKey(c))
                {
                    cur.childrenMap[c] = new TrieNode();
                }
                cur = cur.childrenMap[c];
            }
            cur.isWord = true;
        }
        int wordLength = 0;
        string wordG;
        public bool Search(string word)
        {
            wordLength = word.Length; wordG = word;
            return traverse(0, root);
        }

        private bool traverse(int j, TrieNode root)
        {
            var cur = root;

            for (int i = j; i < wordLength; i++)
            {
                var c = wordG[i];
                if (c.ToString() == ".")
                {
                    foreach (var child in cur.childrenMap)
                    {
                        if (traverse(i + 1, child.Value))                        
                            return true;                         
                    }
                    return false;
                }
                else
                {
                    if (!cur.childrenMap.ContainsKey(c))
                        return false;

                    cur = cur.childrenMap[c];
                }
            }
            return cur.isWord;
        }
    }

}
