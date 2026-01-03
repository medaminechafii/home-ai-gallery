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
const backToTopBtn = document.getElementById("backToTopBtn")

const resultsContainer = document.querySelector(".results")
if (!resultsContainer) {
  console.error("results container not found in DOM")
}
const MEDIA_PATH = "/media/"
const THUMBNAIL_PATH = "/thumbnail/"
/*********************************
 * 2. INITIALIZATION
 *********************************/

// Initialize app on page load
document.addEventListener("DOMContentLoaded", () => {
  initializeApp()
  // Load all media on startup
  loadAllMedia()
})

function initializeApp() {
  searchButton.addEventListener("click", handleSearchClick)
  searchInput.addEventListener("keydown", handleSearchEnter)
  
  // Add event listeners for sliders
  thresholdSlider.addEventListener("input", updateThresholdValue)
  topKSlider.addEventListener("input", updateTopKValue)
  
  // Initialize back to top button
  initializeBackToTop()
}

function initializeBackToTop() {
  // Show/hide back to top button based on scroll position
  window.addEventListener("scroll", () => {
    if (window.pageYOffset > 300) {
      backToTopBtn.classList.add("visible")
    } else {
      backToTopBtn.classList.remove("visible")
    }
  })
  
  // Scroll to top when clicked
  backToTopBtn.addEventListener("click", () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth"
    })
  })
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
  // Scroll to top before searching
  window.scrollTo({ top: 0, behavior: 'smooth' })
  searchFlow(query)
}

function handleSearchEnter(event) {
  if (event.key === "Enter") {
    const query = searchInput.value.trim()
    if (query === "") {
      alert("Please enter a search keyword.")
      return
    }
    // Scroll to top before searching
    window.scrollTo({ top: 0, behavior: 'smooth' })
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

async function loadAllMedia() {
  showLoading("Loading your media collection...")
  try {
    // Use the new all-media endpoint
    const response = await fetch("/api/all-media")
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    hideLoading()
    clearResults()
    
    if (data.media && data.media.length === 0) {
      showEmptyState("No media found. Make sure your media folder contains images or videos.")
      return
    }
    
    // Store all results and render first batch
    currentResults = data.media
    displayedCount = 0
    renderMoreResults()
    
  } catch (error) {
    hideLoading()
    console.error("Load media error:", error)
    showError("Failed to load media collection")
  }
}

/*********************************
 * 6. RENDERING LOGIC
 *********************************/

function clearResults() {
  resultsContainer.innerHTML = ""
  resultsContainer.classList.remove("loading", "empty", "error")
}

function renderResults(mediaList) {
  // Create media grid container
  const gridContainer = document.createElement("div")
  gridContainer.className = "media-grid"
  
  mediaList.forEach((media) => {
    const card = document.createElement("div")
    card.className = "media-card"
    
    if (media.source === "Image") {
      renderImageCard(media, card)
    } else if (media.source === "Video") {
      renderVideoCard(media, card)
    }
    
    gridContainer.appendChild(card)
  })
  
  resultsContainer.appendChild(gridContainer)
}

function renderImageCard(media, card) {
  const img = document.createElement("img")
  img.loading = "lazy"

  if (media.thumbnail) {
    img.src = `${THUMBNAIL_PATH}${media.thumbnail}`
  } else {
    img.src = `${MEDIA_PATH}${encodeURIComponent(media.filename)}`
  }
  
  img.style.cursor = "pointer"
  img.addEventListener("click", () => {
    window.open(`${MEDIA_PATH}${encodeURIComponent(media.filename)}`, "_blank")
  })
  img.alt = media.filename

  const actionsDiv = document.createElement("div")
  actionsDiv.className = "media-actions"

  const downloadBtn = document.createElement("button")
  downloadBtn.className = "download-btn"
  downloadBtn.textContent = "Download"
  downloadBtn.addEventListener("click", () => handleDownload(media))

  actionsDiv.appendChild(downloadBtn)
  card.appendChild(img)
  card.appendChild(actionsDiv)
}

function renderVideoCard(media, card) {
  console.log("Rendering video:", media)
  console.log("Thumbnail hash:", media.thumbnail)

  const videoContainer = document.createElement("div")
  videoContainer.className = "video-container"

  const img = document.createElement("img")
  img.loading = "lazy"
  img.src = `${THUMBNAIL_PATH}${media.thumbnail}`
  img.alt = media.filename
  img.style.cursor = "pointer"
  img.addEventListener("click", () => {
    window.open(`${MEDIA_PATH}${encodeURIComponent(media.filename)}`, "_blank")
  })

  const playIcon = document.createElement("div")
  playIcon.className = "play-icon"
  playIcon.textContent = "▶"

  videoContainer.appendChild(img)
  videoContainer.appendChild(playIcon)
  card.appendChild(videoContainer)

  const actionsDiv = document.createElement("div")
  actionsDiv.className = "media-actions"

  const downloadBtn = document.createElement("button")
  downloadBtn.className = "download-btn"
  downloadBtn.textContent = "Download"
  downloadBtn.addEventListener("click", () => handleDownload(media))

  actionsDiv.appendChild(downloadBtn)
  card.appendChild(actionsDiv)
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
      <p>Searching your media...</p>
      <small>AI is analyzing your images and videos</small>
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
      <p>No results found for your search.</p>
      <small>Try adjusting the similarity threshold or using different keywords.</small>
    </div>
  `
  resultsContainer.classList.add("empty")
}

function showError(errorMessage) {
  resultsContainer.innerHTML = `
    <div class="error-state">
      <p>Something went wrong</p>
      <small>${errorMessage}</small>
      <button class="retry-btn" onclick="retryLastSearch()">Retry Search</button>
    </div>
  `
  resultsContainer.classList.add("error")
}

let lastSearchQuery = ""
let lastSearchParams = {}

function retryLastSearch() {
  if (lastSearchQuery) {
    searchFlow(lastSearchQuery)
  }
}

/*********************************
 * 9. Limit initial Results
 *********************************/


async function searchFlow(keyword) {
  // Store for retry functionality
  lastSearchQuery = keyword
  lastSearchParams = {
    threshold: thresholdSlider.value,
    topK: topKSlider.value
  }
  
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
  
  // Create media grid container if it doesn't exist
  let gridContainer = resultsContainer.querySelector('.media-grid')
  if (!gridContainer) {
    gridContainer = document.createElement("div")
    gridContainer.className = "media-grid"
    resultsContainer.appendChild(gridContainer)
  }
  
  nextBatch.forEach((media) => {
    const card = document.createElement("div")
    card.className = "media-card"
    
    if (media.source === "Image") {
      renderImageCard(media, card)
    } else if (media.source === "Video") {
      renderVideoCard(media, card)
    }
    
    gridContainer.appendChild(card)
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