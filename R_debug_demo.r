# Create a sample data frame
data <- data.frame(
  Name = c("Alice", "Bob", "Charlie", "David", "Eve"),
  Age = c(25, 30, 22, 35, 28),
  Score = c(85, 92, 78, 89, 95)
)

# Custom mean function
custom_mean <- function(data) {
  if (length(data) == 0) {
    return(NULL)
  } else {
    sum_data <- sum(data, na.rm = TRUE)
    num_elements <- length(data) - sum(is.na(data))
    if (num_elements == 0) {
      return(NULL)
    } else {
      return(sum_data / num_elements)
    }
  }
}

# Function to calculate the average age and score
calculate_average <- function(data) {
  avg_age <- custom_mean(data$Age)
  avg_score <- custom_mean(data$Score)
  cat("Average Age:", avg_age, "\n")
  cat("Average Score:", avg_score, "\n")
}

# Call the function to calculate averages
calculate_average(data)



calculate_average(data)
