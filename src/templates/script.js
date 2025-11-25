function search() {
    let query = document.getElementById("query").value;

    fetch("http://127.0.0.1:8000/search", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query: query, topk: 10})
    })
    .then(r => r.json())
    .then(data => {
        let div = document.getElementById("results");
        div.innerHTML = "";

        data.results.forEach(item => {
            div.innerHTML += `
                <div class="result-item">
                    <h2>${item.title}</h2>
                    <p><b>${item.category}</b> • ${item.source}</p>
                    <p>${item.summary}...</p>
                    <p>Similarity: <b>${item.similarity.toFixed(3)}</b></p>
                </div>
            `;
        });
    })
}
