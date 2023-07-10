namespace AliFsmnSharp; 

public class CharTokenizer {
    private readonly string spaceSymbol;
    private readonly HashSet<string> nonLinguisticSymbols;
    private readonly bool removeNonLinguisticSymbols;

    public CharTokenizer(string spaceSymbol = "<space>",
        bool removeNonLinguisticSymbols = false) {
        
        this.spaceSymbol = spaceSymbol;
        nonLinguisticSymbols = new HashSet<string>();
        this.removeNonLinguisticSymbols = removeNonLinguisticSymbols;
    }

    public List<string> Text2Tokens(string line) {
        var tokens = new List<string>();
        while (line.Length != 0) {
            foreach (var w in nonLinguisticSymbols.Where(w => line.StartsWith(w))) {
                if (!removeNonLinguisticSymbols) {
                    tokens.Add(line[..w.Length]);
                }

                line = line[w.Length..];
                break;
            }

            if (line.Any()) continue;
            var t = line[0].ToString();
            if (t == " ") {
                t = spaceSymbol;
            }

            tokens.Add(t);
            line = line[1..];
        }

        return tokens;
    }

    public string Tokens2Text(IEnumerable<string> tokens) {
        return string.Join("", tokens.Select(t => t == spaceSymbol ? " " : t));
    }

    public override string ToString() {
        return
            $"{GetType().Name}(space_symbol=\"{spaceSymbol}\", non_linguistic_symbols=\"{string.Join(", ", nonLinguisticSymbols)}\")";
    }
}