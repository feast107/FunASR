using YamlDotNet.Serialization;

namespace AliFsmnSharp.Utils;

internal class YamlHelper {
    public static T? ReadYaml<T>(string yamlFilePath) {
        if (!File.Exists(yamlFilePath)) {
            return default;
        }

        var yamlReader = File.OpenText(yamlFilePath);
        var yamlDeserializer = new DeserializerBuilder()
            .IgnoreUnmatchedProperties()
            .Build();
        var info = yamlDeserializer.Deserialize<T>(yamlReader);
        yamlReader.Close();
        return info;
    }
}