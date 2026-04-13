# CopilotJ Development

This project contains three parts:

1. **Agent**: The core logic that understands user requests, interacts with AI models, and orchestrates tasks. It
   communicates with the ImageJ plugin via the Bridge server to control ImageJ.
2. **ImageJ plugin**: A Java-based plugin running within ImageJ. It listens for commands from the Agent (relayed by the
   Bridge server), executes these commands using ImageJ's APIs, and sends results or state changes back.
3. **Bridge server**: A communication hub that facilitates interaction between the Python-based Agent and the
   Java-based ImageJ plugin. It typically uses WebSockets to relay messages and commands.

Run the bridge server:

```bash
python -m copilotj.server
```

The server is designed to be stable due to its minimal and focused design.

Run the plugin (with Fiji):

```bash
cd plugin && \
  mvn exec:java -D"exec.mainClass=copilotj.DefaultCopilotJBridgeService" -D"ij.debug=true"
```

Run the multiagent:

```bash
python -m copilotj.multiagent
```

If you make changes to the Java code, restart the plugin to apply the updates.

Once everything is running, look for the message `Bridge WebSocket connection established` in the server console to
confirm that the connection is active.

### Manual Plugin Deployment

If you prefer to manually deploy the plugin to your Fiji installation:

1. **Build and install the plugin and its dependencies:**
   Navigate to the `plugin` directory and build the package:
   ```bash
   cd plugin && mvn clean install -Dscijava.app.directory=/path/to/Fiji
   ```
   where `/path/to/Fiji` is the file path to your Fiji installation folder.
   This will create the plugin JAR file (e.g., `CopilotJBridge-1.0.0.jar`)
   in the `plugin/target/` directory, then copy it along with all of its
   dependency JAR files into the specified Fiji installation.

   Note that Fiji comes bundled with many of CopilotJ's dependencies, but the
   [SciJava infrastructure](https://github.com/scijava/scijava-maven-plugin/)
   takes care to keep only the newer version of each dependency JAR when
   copying them.

2. **Restart Fiji:**
   After the build is complete, restart Fiji for the changes to take effect.
