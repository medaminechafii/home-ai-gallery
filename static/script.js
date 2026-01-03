/*********************************
 * GLOBAL STATE
 *********************************/
let currentResults = []
let displayedCount = 0   
const RESULT_PER_PAGE = 20
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

const thresholdSlider = document.getElementById("thresholdSlider")
const thresholdValue = document.getElementById("thresholdValue")
const topKSlider = document.getElementById("topKSlider")
const topKValue = document.getElementById("topKValue")

const resultsContainer = document.querySelector(".results")
if (!resultsContainer) {
  console.error("results container not found in DOM")
}
const MEDIA_PATH = "/media/"
const THUMBNAIL_PATH = "/thumbnail/"
/*********************************
 * 2. INITIALIZATION
 *********************************/

function initializeApp() {
  searchButton.addEventListener("click", handleSearchClick)
  searchInput.addEventListener("keydown", handleSearchEnter)
  
  // Add event listeners for sliders
  thresholdSlider.addEventListener("input", updateThresholdValue)
  topKSlider.addEventListener("input", updateTopKValue)
}

function updateThresholdValue() {
  thresholdValue.textContent = thresholdSlider.value
}

function updateTopKValue() {
  topKValue.textContent = topKSlider.value
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
  const threshold = parseFloat(thresholdSlider.value)
  const topK = parseInt(topKSlider.value)
  
  const response = await fetch("/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ 
      query: keyword,
      score_threshold: threshold,
      top_k: topK
    }),
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
    } else if (media.source === "Video") {
      renderVideoCard(media)
    }
  })
}

function renderImageCard(media) {
  const card = document.createElement("div")
  card.className = "media-card"

  const img = document.createElement("img")

  img.loading = "lazy"

  if (media.thumbnail){
    img.src = `${THUMBNAIL_PATH}${media.thumbnail}`
  }
  else{
    img.src = `${MEDIA_PATH}${encodeUIOComponent(media.filename)}`
  }
  img.style.cursor = "pointer"
  img.addEventListener("click", () => {
    window.open(`${MEDIA_PATH}${encodeURIComponent(media.filename)}`, "_blank")
  })
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
  console.log("Rendering video:", media)
  console.log("Thumbnail hash:", media.thumbnail)

  const card = document.createElement("div")
  card.className = "media-card"

  const videoContainer = document.createElement("div")
  videoContainer.className = "video-container"
  const thumbnail = document.createElement("img")
  if (media.thumbnail){
    thumbURL = `${THUMBNAIL_PATH}${media.thumbnail}`
    console.log("Thumbnail URL:", thumbURL)
    thumbnail.src = thumbURL
  }
  else{
    console.log("No thumbnail, using media file")
    thumbURL = `${MEDIA_PATH}${encodeURIComponent(media.filename)}`
    thumbnail.src = thumbURL
  }
  thumbnail.className = "video-thumbnail"

  const playButton = document.createElement("div")
  playButton.className = "play-button"
  playButton.innerHTML = "▶"
  videoContainer.addEventListener("click", () => {
    videoContainer.innerHTML = ""
    video = document.createElement("video")
    video.src = `${MEDIA_PATH}${encodeURIComponent(media.filename)}`
    video.controls = true
    video.autoplay = true
    video.className = "video-player"
    videoContainer.appendChild(video)
  })

  videoContainer.appendChild(thumbnail)
  videoContainer.appendChild(playButton)

  const scoreBadge = document.createElement("div")
  scoreBadge.className = "score-badge"
  scoreBadge.textContent = `${(media.score * 100).toFixed(1)}%`

  const actionsDiv = document.createElement("div")
  actionsDiv.className = "media-actions"

  const downloadBtn = document.createElement("button")
  downloadBtn.textContent = "Download"
  downloadBtn.addEventListener("click", () => handleDownload(media))

  actionsDiv.appendChild(downloadBtn)
  card.appendChild(videoContainer)
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
/*********************************
 * 9. Limit initial Results
 *********************************/


async function searchFlow(keyword) {
  showLoading()
  try{
    const results = await fetchSearchResults(keyword)
    hideLoading()
    clearResults()
    if (results.length === 0) {
      showEmptyState()
      return
    }
    currentResults = results
    displayedCount = 0
    renderMoreResults()
  }
  catch (error) {
    console.error("Search error:", error)
    hideLoading()
    showError(error.message)
  }
}
function renderMoreResults() {
  hideLoadMoreButton()
  const nextBatch = currentResults.slice(displayedCount, displayedCount + RESULT_PER_PAGE)
  nextBatch.forEach((media) => {
    if (media.source === "Image") {
      renderImageCard(media)
    } else if (media.source === "Video") {
      renderVideoCard(media)
    }
  })
  displayedCount += nextBatch.length 
  if(displayedCount < currentResults.length){
    showLoadMoreButton()
  }
  else{
    hideLoadMoreButton()
  }
}
function showLoadMoreButton() {
  let loadMoreBtn = document.getElementById("load-more-btn")
  if (!loadMoreBtn) {
    loadMoreBtn = document.createElement("button")
    loadMoreBtn.id = "load-more-btn"
    loadMoreBtn.textContent = "Load More"
    loadMoreBtn.addEventListener("click", renderMoreResults)
    resultsContainer.appendChild(loadMoreBtn)
  }
  else{
    loadMoreBtn.textContent = `Load More (${currentResults.length - displayedCount} remaining)`
    resultsContainer.appendChild(loadMoreBtn)
  }
}

function hideLoadMoreButton() {
  const loadMoreBtn = document.getElementById("load-more-btn")
  if (loadMoreBtn) {
    loadMoreBtn.remove()
  }
}