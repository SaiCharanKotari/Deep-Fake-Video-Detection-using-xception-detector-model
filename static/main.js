let uploadedFile = "";

    async function upload() {
      const fileInput = document.getElementById("videoFile");
      const file = fileInput.files[0];
      if (!file) {
        alert("Please select a file first!");
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/upload", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      uploadedFile = data.filename;
      alert("Uploaded: " + uploadedFile);
    }

    async function run() {
      if (!uploadedFile) {
        alert("Please upload a file first!");
        return;
      }

      const res = await fetch(`/run/${uploadedFile}`);
      const data = await res.json();

      document.getElementById("result").innerText = data.message;

      // Show CSV results if available
      if (data.results && data.results.length > 0) {
        let output = "";
        data.results.forEach(row => {
          output += JSON.stringify(row) + "\n";
        });
        document.getElementById("csvOutput").innerText = output;
      }
      else {
        document.getElementById("csvOutput").innerText = "No CSV results found.";
      }
    }