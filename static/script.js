/*********************************
 * 1. DOM REFERENCES
 *********************************/

console.log("script.js loaded")
const searchInput = document.getElementById("searchInput")
if (!searchInput) {
  console.error("Search input not found in DOM")
}

const searchButton = document.getElementById("searchBtn")
if (!searchButton) {
  console.error("search button not found in DOM")
}

const resultsContainer = document.querySelector(".results")
if (!resultsContainer) {
  console.error("results container not found in DOM")
}

/*********************************
 * 2. INITIALIZATION
 *********************************/

function initializeApp() {
  searchButton.addEventListener("click", handleSearchClick)
  searchInput.addEventListener("keydown", handleSearchEnter)
}

initializeApp()

/*********************************
 * 3. EVENT HANDLERS
 *********************************/

function handleSearchClick() {
  const query = searchInput.value.trim()
  if (query === "") {
    alert("Please enter a search keyword.")
    return
  }
  searchFlow(query)
}

function handleSearchEnter(event) {
  if (event.key === "Enter") {
    const query = searchInput.value.trim()
    if (query === "") {
      alert("Please enter a search keyword.")
      return
    }
    searchFlow(query)
  }
}

/*********************************
 * 4. CORE SEARCH FLOW
 *********************************/

async function searchFlow(keyword) {
  showLoading()
  try {
    const results = await fetchSearchResults(keyword)
    hideLoading()
    clearResults()

    if (results.length === 0) {
      showEmptyState()
      return
    }

    renderResults(results)
  } catch (error) {
    hideLoading()
    console.error("Search error:", error)
    showError(error.message)
  }
}

/*********************************
 * 5. API COMMUNICATION
 *********************************/

async function fetchSearchResults(keyword) {
  const response = await fetch("/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query: keyword }),
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(`HTTP error! status: ${errorData.status}`)
  }

  const data = await response.json()
  console.log("API response data:", data)
  return data.results
}

/*********************************
 * 6. RENDERING LOGIC
 *********************************/

function clearResults() {
  resultsContainer.innerHTML = ""
  resultsContainer.classList.remove("loading", "empty", "error")
}

function renderResults(mediaList) {
  mediaList.forEach((media) => {
    if (media.source === "Image") {
      renderImageCard(media)
    } else if (media.source === "Video Frame") {
      renderVideoCard(media)
    }
  })
}

function renderImageCard(media) {
  const card = document.createElement("div")
  card.className = "media-card"

  const img = document.createElement("img")
  img.src = `/media/${media.filename}`
  img.alt = media.filename

  const scoreBadge = document.createElement("div")
  scoreBadge.className = "score-badge"
  scoreBadge.textContent = `${(media.score * 100).toFixed(1)}%`

  const actionsDiv = document.createElement("div")
  actionsDiv.className = "media-actions"

  const downloadBtn = document.createElement("button")
  downloadBtn.textContent = "Download"
  downloadBtn.addEventListener("click", () => handleDownload(media))

  actionsDiv.appendChild(downloadBtn)
  card.appendChild(img)
  card.appendChild(scoreBadge)
  card.appendChild(actionsDiv)
  resultsContainer.appendChild(card)
}

function renderVideoCard(media) {
  const card = document.createElement("div")
  card.className = "media-card"

  const video = document.createElement("video")
  video.src = `/media/${media.filename}`
  video.controls = true

  const scoreBadge = document.createElement("div")
  scoreBadge.className = "score-badge"
  scoreBadge.textContent = `${(media.score * 100).toFixed(1)}%`

  const actionsDiv = document.createElement("div")
  actionsDiv.className = "media-actions"

  const downloadBtn = document.createElement("button")
  downloadBtn.textContent = "Download"
  downloadBtn.addEventListener("click", () => handleDownload(media))

  actionsDiv.appendChild(downloadBtn)
  card.appendChild(video)
  card.appendChild(scoreBadge)
  card.appendChild(actionsDiv)
  resultsContainer.appendChild(card)
}

/*********************************
 * 7. DOWNLOAD HANDLING
 *********************************/

function handleDownload(media) {
  const filename = media.filename

  const link = document.createElement("a")
  link.href = "/media/" + filename
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

/*********************************
 * 8. UI STATE HELPERS
 *********************************/

function showLoading() {
  resultsContainer.innerHTML = `
        <div class="loading-spinner">
            <div class="spinner"></div>
            <p>Searching...</p>
        </div>
    `
  resultsContainer.classList.add("loading")
}

function hideLoading() {
  resultsContainer.classList.remove("loading")
}

function showEmptyState() {
  resultsContainer.innerHTML = `
        <div class="empty-state">
            <p>No results found. Try a different search term.</p>
        </div>
    `
  resultsContainer.classList.add("empty")
}

function showError(message) {
  resultsContainer.innerHTML = `
        <div class="error-state">
            <p>Error: ${message}</p>
            <p style="margin-top: 8px; font-size: 14px;">Please try again.</p>
        </div>
    `
  resultsContainer.classList.add("error")
}
