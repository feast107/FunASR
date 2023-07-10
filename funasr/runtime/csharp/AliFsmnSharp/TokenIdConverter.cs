namespace AliFsmnSharp; 

public class TokenIdConverter {
    private readonly List<string> tokenList;

    public TokenIdConverter(List<string> tokenList) {
        this.tokenList = tokenList;
    }

    public string[] Ids2Tokens(IEnumerable<int> integers) {
        return integers.Select(i => tokenList[i]).ToArray();
    }
}